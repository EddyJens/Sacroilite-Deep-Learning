package Deeplearning.mestrado;

import java.io.IOException;

import javax.swing.text.AbstractDocument.LeafElement;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
/*
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/transferlearning/vgg16/TransferLearning.md
 * */
public class TransferLearningSacroileite {
	private static Logger log = LoggerFactory.getLogger(App.class);
	
	
	public static ComputationGraph googLeNet(int numClass, int seed, int iterations){
		ZooModel zooModel = new GoogLeNet (numClass, seed, iterations);
		int[][] shape = {{1, 100, 100}};
		zooModel.setInputShape(shape);
		/*****PRETRAINED WEIGTHS*****/
		try {
			
			ComputationGraph  net = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
			/*ComputationGraph googLeNetTransfer = new TransferLearning.GraphBuilder(net) //the specified layer and below are "frozen"
					 	.setFeatureExtractor("fc1") //the specified layer and below are "frozen"   
					 	.removeVertexKeepConnections("5b-depthconcat1") //replace the functionality of the final vertex
			            .addLayer("5b-depthconcat1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nIn(1024).nOut(2).activation(Activation.SOFTMAX).build(), "fc1")
			            .build();*/
			//output = ComputationGraph(net.conf());
			return net;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static MultiLayerNetwork  alexNet(int numClass, int seed, int iterations){
		ZooModel zooModel = new AlexNet (numClass, seed, iterations);
		int[][] shape = {{1, 100, 100}};
		zooModel.setInputShape(shape);
		/*****PRETRAINED WEIGTHS*****/
		try {
			
			MultiLayerNetwork  net = (MultiLayerNetwork) zooModel.init();
			
			return net;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static MultiLayerNetwork leNet(int numClass, int seed, int iterations){
		ZooModel zooModel = new LeNet (numClass, seed, iterations);
		int[][] shape = {{1, 100, 100}};
		zooModel.setInputShape(shape);
		try {
			
			MultiLayerNetwork  net = (MultiLayerNetwork) zooModel.init();

			return net;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
}
