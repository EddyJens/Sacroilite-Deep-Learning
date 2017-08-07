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
	//modelType available MLP1->1, LENET->2, ALEXNET->3, VGG16->4, simpleCNN->5
	protected static int modelType = 3;
	protected static int batchSize = 10;//20
	protected static int numEpochs = 10;//50
	protected static int iterations = 1;
	protected static double learningRate = 0.001;
	
	
    public static void main( String[] args ) throws IOException
    {
    	BasicConfigurator.configure();
    	//Alterar de acordo com seu diretorio
    	File trainData = new File("C:\\Users\\ejrza_000\\Desktop\\concatenadas28.07\\treinamentoConcatenado");
        //File trainData = new File("C:\\Users\\ejrza_000\\Downloads\\mestrado\\mestrado\\treinamento");
        File testData = new File("C:\\Users\\ejrza_000\\Desktop\\concatenadas28.07\\testeConcatenado");
        //File testData = new File("C:\\Users\\ejrza_000\\Downloads\\mestrado\\mestrado\\teste");
        
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
                model = DefaultNetworks.mlp(seed, iterations, learningRate, height, width, channels, outputNum);
                break;
            case 2:
                model = DefaultNetworks.leNet(seed, iterations, learningRate, height, width, channels, outputNum);
                break;
            case 3:
                model = DefaultNetworks.alexNet(seed, iterations, learningRate, height, width, channels, outputNum);
                break;
            case 4:
                model = DefaultNetworks.VGG16(seed, iterations, learningRate, height, width, channels, outputNum);
                break;
            case 5:
                model = DefaultNetworks.simpleCNN(seed, iterations, learningRate, height, width, channels, outputNum);
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
        for(int i = 0; i < numEpochs; i++){
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

}
