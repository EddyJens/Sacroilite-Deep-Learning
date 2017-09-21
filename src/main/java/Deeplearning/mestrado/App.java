package Deeplearning.mestrado;

import org.deeplearning4j.api.storage.StatsStorage;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
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
	protected static int height = 42;
	protected static int width = 256;
	protected static int channels = 1;
	protected static int outputNum = 2;
	protected static int batchSize = 5;//20
	protected static int numEpochs = 1;//50
	protected static int iterations = 1;
	
    public static void main( String[] args ) throws IOException
    {
    	BasicConfigurator.configure();
    	//Alterar de acordo com seu diretorio
    	File trainData = new File("C:\\Users\\mcfal\\Desktop\\PermutaResized\\Treino");
        File testData = new File("C:\\Users\\mcfal\\Desktop\\PermutaResized\\Teste");

        //Configuring UI
        //Initialize the user interface backend
       
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
 
        FileSplit train = new FileSplit(trainData);
        FileSplit test = new FileSplit(testData);

        // Extract the parent path as the image label

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng,BaseImageLoader.ALLOWED_FORMATS, labelMaker); //shuffle data
        
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        InputSplit[] trainDataShuffled = train.sample(pathFilter, 1,0); //shuffle data
        recordReader.initialize(trainDataShuffled[0]);

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        // Scale pixel values to 0-1

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);

        log.info("Build model....");


        MultiLayerNetwork model;
      
        model = TransferLearningSacroileite.leNet(2, (int)seed, iterations, width, height);

        model.init();

        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // The Score iteration Listener will log
        // output to show how well the network is training
    	model.setListeners(new StatsListener(statsStorage));

        log.info("*****TRAIN MODEL********");
        for(int i = 0; i < numEpochs; i++){
        	log.info("******EPOCH = "+i+"*****");
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
