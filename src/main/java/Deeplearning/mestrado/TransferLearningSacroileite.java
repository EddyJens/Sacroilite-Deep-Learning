package Deeplearning.mestrado;

import java.io.IOException;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.AlexNet;
import org.deeplearning4j.zoo.model.GoogLeNet;
/*
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/transferlearning/vgg16/TransferLearning.md
 * */
public class TransferLearningSacroileite {
	private static Logger log = LoggerFactory.getLogger(App.class);
	
	public static ComputationGraph googLeNetImageNet(int numClass, int seed, int iterations){
		ZooModel zooModel = new GoogLeNet(numClass, seed, iterations);
		ComputationGraph output = null;
		/*****PRETRAINED WEIGTHS*****/
		try {
			Model net = zooModel.initPretrained(PretrainedType.IMAGENET);
			//output = ComputationGraph(net.conf());
			return output;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
}
