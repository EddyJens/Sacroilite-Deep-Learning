package Deeplearning.mestrado;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DefaultNetworks {
	
	/* 
	 * rede com apenas uma camada densa de neuronios
	 * retirada de um dos exemplos do dl4j
	 * tem taxa de acertos elevada para pequenas bases de dados
	 * 
	 * @param seed
	 * @param iterations
	 * @param learningRate
	 * @param height
	 * @param width 
	 * @param channels
	 * @param outputNum
	 * @ return MultilayerNetwork with Multilayer Perceptron configurations
	 * 
	 * Source: Insert article here
	 * */
	public static MultiLayerNetwork mlp1(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum) {

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

	/*
	 * rede com duas camadas densas de neuronios 
	 * feita com o auxilio da rede anterior
	 * 
	 * @param seed
	 * @param iterations
	 * @param learningRate
	 * @param height
	 * @param width 
	 * @param channels
	 * @param outputNum
	 * @return MultilayerNetwork with Multilayer Perceptron configuration
	 * 
	 * Source: Insert article here
	 * */
    public static MultiLayerNetwork mlp2(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum) {

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

	/*
	 * arquitetura lenet, retirada do exemplo animals da framework
 	 * todos os valores dos hyperparametros sao padrao 
	 *
	 * @param seed
	 * @param iterations
	 * @param learningRate
	 * @param height
	 * @param width 
	 * @param channels
	 * @param outputNum
	 * @return MultilayernNetwork with LeNet configuration
	 * Source: Insert article here
	 * */
    public static MultiLayerNetwork leNet(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum) {

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
            .layer(0, Layers.convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
            .layer(1, Layers.maxPool("maxpool1", new int[]{2, 2}))
            .layer(2, Layers.conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
            .layer(3, Layers.maxPool("maxool2", new int[]{2, 2}))
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
    
	/*
	 * arquitetura alexnet, retirada do exemplo animals da framework
 	 * todos os valores dos hyperparametros sao padrao 
	 *
	 * @param seed
	 * @param iterations
	 * @param learningRate
	 * @param height
	 * @param width 
	 * @param channels
	 * @param outputNum
	 * @return MultilayerNetwork with AlexNet configuration
	 * 
	 * Source: Insert article here
	 * */
    public static MultiLayerNetwork alexNet(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum){
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
                .layer(0, Layers.convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, Layers.maxPool("maxpool1", new int[]{3,3}))
                .layer(3, Layers.conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, Layers.maxPool("maxpool2", new int[]{3,3}))
                .layer(6, Layers.conv3x3("cnn3", 384, 0))
                .layer(7, Layers.conv3x3("cnn4", 384, nonZeroBias))
                .layer(8, Layers.conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, Layers.maxPool("maxpool3", new int[]{3,3}))
                .layer(10, Layers.fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, Layers.fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
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
    
	/*
	 * @param seed
	 * @param iterations
	 * @param learningRate
	 * @param height
	 * @param width 
	 * @param channels
	 * @param outputNum
	 * @return MultilayerNetwork with singleConv configuration
	 * 
	 * TODO metodo todo
	 * Source: Insert article here
	 * */
    public static MultiLayerNetwork singleConv(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum){

        return null;
    }
    
	/*
	 * @param seed
	 * @param iterations
	 * @param learningRate
	 * @param height
	 * @param width 
	 * @param channels
	 * @param outputNum
	 * @return MultilayerNetwork with fullyConv configuration
	 * 
	 * TODO metodo todo
	 * Source: Insert article here
	 * */
    public static MultiLayerNetwork fullyConv(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum){
        return null;
    }
}
