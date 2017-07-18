package Deeplearning.mestrado;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
/*
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/transferlearning/vgg16/TransferLearning.md
 * */
public class TransferLearningSacroileite {
	private static Logger log = LoggerFactory.getLogger(App.class);
	
	public ComputationGraph vgg16(int numClasses) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException{
		//Importing VGG16
		log.info("**********IMPORTING VGG16***********");
		TrainedModelHelper modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16);
		ComputationGraph vgg16 = modelImportHelper.loadModel();
		
		//Set up a fine-tune configuration
		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
	            .learningRate(5e-5)
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .updater(Updater.NESTEROVS)
	            .seed(42)
	            .build();
		
		//Build new models based on VGG16
		log.info("**********Building new Model Based on VGG16***********");
		//Modifying only the last layer, keeping other frozen
		//The final layer of VGG16 does a softmax regression on the 1000 classes in ImageNet.
		//We modify the very last layer to give predictions for 2 classes keeping the other layers frozen.
		ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
		 		.fineTuneConfiguration(fineTuneConf)
		            	.setFeatureExtractor("fc2")
		            	.removeVertexKeepConnections("predictions") 
		            	.addLayer("predictions", 
				  	new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
		                    		.nIn(4096).nOut(numClasses)
		                    		.weightInit(WeightInit.XAVIER)
		                    		.activation(Activation.SOFTMAX).build(), "fc2")
		         .build();
		
		//Saving “featurized” datasets and training with them.
		//TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Transfer);
		
		return vgg16Transfer; //return the loaded model with the last layer changed
	}
}
