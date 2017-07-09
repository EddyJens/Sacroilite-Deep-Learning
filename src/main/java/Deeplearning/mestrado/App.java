package Deeplearning.mestrado;

/**
 * Referencia https://search.maven.org/#search%7Cga%7C2%7Cdeeplearning4j
 * Pom file importa dependencias Maven, veja em: https://github.com/deeplearning4j/deeplearning4j
 *
 */
 //importando tudo de tudo pq nunca se sabe...
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

//import org.apache.*;
import org.apache.log4j.BasicConfigurator;

public class App {
	
	/******************Variaveis Globais*******************/
	private static Logger log = LoggerFactory.getLogger(App.class);
	protected static long seed = 42;
	protected static Random rng = new Random(seed);
	protected static int height;
	protected static int width;
	protected static int channels = 1;
	protected static int outputNum = 2;
	//modelType available MLP1->1, MLP2->2, LENET->3, ALEXNET->4, SINGLECONV->5, CONVFULLY->6
	protected static int modelType = 4;
	protected static int batchSize = 20;
	protected static int numEpochs = 50;
	protected static int iterations = 1;
	protected static double learningRate = 0.001;
	
	
    public static void main( String[] args ) throws IOException
    {
    	BasicConfigurator.configure();
    	//Alterar de acordo com seu diretorio
    	File trainData = new File("C:\\Users\\ejrza_000\\Downloads\\mestrado\\mestrado\\teste");
        File testData = new File("C:\\Users\\ejrza_000\\Downloads\\mestrado\\mestrado\\treinamento");
        
        //dimensao de entrada
        height = 100;
        width = 100;
        
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,rng);
        FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,rng);

        // Extract the parent path as the image label

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name

        recordReader.initialize(train);
        //recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        // Scale pixel values to 0-1

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);

        log.info("Build model....");


        MultiLayerNetwork model;
        switch (modelType) {
            case 1:
                model = MLP1();
                break;
            case 2:
                model = MLP2();
                break;
            case 3:
                model = LENET();
                break;
            case 4:
                model = ALEXNET();
                break;
            case 5:
                model = SINGLECONV();
                break;
            case 6:
                model = CONVFULLY();
                break;

            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }
        model.init();

        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // The Score iteration Listener will log
        // output to show how well the network is training
        model.setListeners(new ScoreIterationListener(2));//was 10

        log.info("*****TRAIN MODEL********");
        for(int i = 0; i<numEpochs; i++){
            model.fit(dataIter);
        }

        log.info("******EVALUATE MODEL******");

        recordReader.reset();

        // The model trained on the training dataset split
        // now that it has trained we evaluate against the
        // test data of images the network has not seen

        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        /*
        log the order of the labels for later use
        In previous versions the label order was consistent, but random
        In current verions label order is lexicographic
        preserving the RecordReader Labels order is no
        longer needed left in for demonstration
        purposes
        */
        //log.info(recordReader.getLabels().toString());

        // Create Eval object with 2 possible classes
        Evaluation eval = new Evaluation(outputNum);

        // Evaluate the network
        while(testIter.hasNext()){
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            // Compare the Feature Matrix from the model
            // with the labels from the RecordReader
            eval.eval(next.getLabels(),output);

        }
        log.info(eval.stats());
    }

    /*
     * Definicao das layers.
     */
    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }
    private static SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }
    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }
    private static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }
    private static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }
    
  //modelo single layer - perceptron
    public static MultiLayerNetwork MLP1() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(iterations)
            .learningRate(learningRate)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(height * width * channels)
                .nOut(100)//was 100
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(100)//was 100
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .pretrain(false).backprop(true)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);

    }

    //modelo two layers (MLP)
    public static MultiLayerNetwork MLP2() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(learningRate)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(height * width * channels)
                .nOut(100)//was 100
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nIn(100)
                .nOut(100)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(100)//was 100
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .pretrain(false).backprop(true)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);

    }
    //modelo LeNet
    public static MultiLayerNetwork LENET() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(5)
            .regularization(false).l2(0.005) // tried 0.0001, 0.0005
            .activation(Activation.RELU)
            .learningRate(0.000001) // tried 0.00001, 0.00005, 0.000001
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.RMSPROP).momentum(0.9)
            .list()
            .layer(0, convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
            .layer(1, maxPool("maxpool1", new int[]{2, 2}))
            .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
            .layer(3, maxPool("maxool2", new int[]{2, 2}))
            .layer(4, new DenseLayer.Builder().nOut(500).build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true).pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);

    }
    
    //modelo alexnet
    public static MultiLayerNetwork ALEXNET(){
            /**
             * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
             * and the imagenetExample code referenced.
             * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
             **/

            double nonZeroBias = 1;
            double dropOut = 0.5;

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{3,3}))
                .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3,3}))
                .layer(6,conv3x3("cnn3", 384, 0))
                .layer(7,conv3x3("cnn4", 384, nonZeroBias))
                .layer(8,conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3,3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .name("output")
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

            return new MultiLayerNetwork(conf);

    }
    
    //modelo single convolution
    public static MultiLayerNetwork SINGLECONV(){

        return null;
    }
    
    //modelo conv + fully connected
    public static MultiLayerNetwork CONVFULLY(){
        return null;
    }

}
