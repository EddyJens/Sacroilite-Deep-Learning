package Deeplearning.mestrado;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import javax.swing.text.AbstractDocument.LeafElement;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationRationalTanh;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.collection.immutable.HashMap;

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
	
	public static MultiLayerNetwork  alexNet(int numClass, int seed, int iterations,int width, int height){
		ZooModel zooModel = new AlexNet (numClass, seed, iterations);
		int[][] shape = {{1, width, height}};
		zooModel.setInputShape(shape);
		/*****PRETRAINED WEIGTHS*****/
		try {
			
			MultiLayerNetwork  net = (MultiLayerNetwork) zooModel.init();
			FineTuneConfiguration finetune = new FineTuneConfiguration.Builder()
					.learningRate(0.000001)
					.momentum(0.3)
					.build();
			
			return new TransferLearning.Builder(net).fineTuneConfiguration(finetune).build();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static MultiLayerNetwork  vgg16(int numClass, int seed, int iterations,int width, int height){
		ZooModel zooModel = new VGG16 (numClass, seed, iterations);
		int[][] shape = {{1, width, height}};
		zooModel.setInputShape(shape);
		try {
			
			MultiLayerNetwork  net = (MultiLayerNetwork) zooModel.init();
			FineTuneConfiguration finetune = new FineTuneConfiguration.Builder()
					.learningRate(1e-3)
					.updater(Updater.NESTEROVS)
					.build();
			
			return new TransferLearning.Builder(net).fineTuneConfiguration(finetune).build();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static MultiLayerNetwork leNet(int numClass, int seed, int iterations, int width, int height){
		ZooModel zooModel = new LeNet (numClass, seed, iterations);
		int[][] shape = {{1, width, height}};
		zooModel.setInputShape(shape);
		try {
			Map<Integer, Double> lrSchedule = new java.util.HashMap();
			
			lrSchedule.put(0, 1e-3);
			MultiLayerNetwork  net = (MultiLayerNetwork) zooModel.init();
			net.summary();
			FineTuneConfiguration finetune = new FineTuneConfiguration.Builder()
					.learningRatePolicy(LearningRatePolicy.Schedule)
					.learningRateSchedule(lrSchedule)
					.updater(Updater.NESTEROVS)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.activation(Activation.SIGMOID)
					.build();
			
			return new TransferLearning.Builder(net).fineTuneConfiguration(finetune).build();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
	
	/*
	 * Configuração padrão da LeNet:
	 * Learning Rate Inicial 1e-3
	 * Inicialização dos pesos Distribuição media 0 dp 0.01
	 * Ativação sigmoid
	 * Learning Rate Score Based Decay 1e-1
	 * Otimização Gradiente Descendente Estocastico
	 * Input -> Conv -> SubSamp -> Conv -> SubSamp -> Dense -> Dense -> Output
	 * Backprop TRUE
	 * SOURCE: https://github.com/deeplearning4j/dl4j-model-z/blob/master/src/main/java/org/deeplearning4j/LeNet.java
	 * 
	 * @param train DataSetIterator contendo os dados de treino
	 * @param test DataSetIterator contendo os dados de teste
	 * @param width largura da imagem
	 * @param height altura da imagem
	 * @param seed Semente usada para gerar os valores pseudoaleatorios, importante para garantir consistencia nos testes
	 * @param iterations numero de iteracoes 
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
		public static MultiLayerNetwork earlyStopLeNet(DataSetIterator train, DataSetIterator test, int width, int height, int seed, int iterations){
			ZooModel zooModel = new LeNet (2, seed, iterations);
			int[][] shape = {{1, width, height}};
			zooModel.setInputShape(shape);
			MultiLayerNetwork  net = (MultiLayerNetwork) zooModel.init();
	
			
			EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver();
			EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder()
					//.epochTerminationConditions(new MaxEpochsTerminationCondition(30))
					.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
					.scoreCalculator(new DataSetLossCalculator(test, true))
			        .evaluateEveryNEpochs(1)
					.modelSaver(saver)
					.build();
	
			EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,net,train);
	
			//Conduct early stopping training:
			EarlyStoppingResult result = trainer.fit();
	
			//Print out the results:
			System.out.println("Termination reason: " + result.getTerminationReason());
			System.out.println("Termination details: " + result.getTerminationDetails());
			System.out.println("Total epochs: " + result.getTotalEpochs());
			System.out.println("Best epoch number: " + result.getBestModelEpoch());
			System.out.println("Score at best epoch: " + result.getBestModelScore());
	
			//Get the best model:
			MultiLayerNetwork bestModel = (MultiLayerNetwork) result.getBestModel();
			if(bestModel == null){
				System.out.println("BestModel is null");
			}
			return bestModel;
			
		}
}
