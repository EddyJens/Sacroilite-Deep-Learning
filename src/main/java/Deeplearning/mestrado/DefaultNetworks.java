package Deeplearning.mestrado;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.LeNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
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
	public static MultiLayerNetwork mlp(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum) {

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
	 * LeNet was an early promising achiever on the ImageNet dataset
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
	 * Source: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
	 * @author kepricon
     * @author Justin Long (crockpotveggies)
	 * */
    public static MultiLayerNetwork leNet(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum) {

        int[] inputShape = new int[] {channels, height, width};
        WorkspaceMode workspaceMode;
        workspaceMode = WorkspaceMode.SEPARATE;


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode).seed(seed).iterations(iterations)
                .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new AdaDelta())
                .regularization(false).convolutionMode(ConvolutionMode.Same).list()
                // block 1
                .layer(0, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}).name("cnn1")
                        .nIn(inputShape[0]).nOut(20).activation(Activation.RELU).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2},
                        new int[] {2, 2}).name("maxpool1").build())
                // block 2
                .layer(2, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}).name("cnn2").nOut(50)
                        .activation(Activation.RELU).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2},
                        new int[] {2, 2}).name("maxpool2").build())
                // fully connected
                .layer(4, new DenseLayer.Builder().name("ffn1").activation(Activation.RELU).nOut(500).build())
                // output
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).name("output")
                        .nOut(outputNum).activation(Activation.SOFTMAX) // radial basis function required
                        .build())
                .setInputType(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                .backprop(true).pretrain(false).build();

        return new MultiLayerNetwork(conf);

    }
    
	/*
     * AlexNet
     *
     * Dl4j's AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
     * and the imagenetExample code referenced.
     *
     * Model is built in dl4j based on available functionality and notes indicate where there are gaps waiting for enhancements.
     *
     * Bias initialization in the paper is 1 in certain layers but 0.1 in the imagenetExample code
     * Weight distribution uses 0.1 std for all layers in the paper but 0.005 in the dense layers in the imagenetExample code
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
	 * Source:
	 * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
     * https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt
     *
	 * */
    public static MultiLayerNetwork alexNet(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum){

        int[] inputShape = new int[] {channels, height, width};
        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).learningRate(learningRate).biasLearningRate(1e-2 * 2).regularization(true)
                .convolutionMode(ConvolutionMode.Same)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .dropOut(0.5).l2(5 * 1e-4).miniBatch(false)
                .list().layer(0,
                        new ConvolutionLayer.Builder(new int[] {11, 11}, new int[] {4, 4},
                                new int[] {2, 2}).name("cnn1")
                                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                                .convolutionMode(ConvolutionMode.Truncate)
                                .nIn(inputShape[0]).nOut(64).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                        new int[] {2, 2}, new int[] {1, 1}).convolutionMode(ConvolutionMode.Truncate)
                        .name("maxpool1").build())
                .layer(2, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {2, 2}, new int[] {2, 2}) // TODO: fix input and put stride back to 1,1
                        .convolutionMode(ConvolutionMode.Truncate).name("cnn2")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(192)
                        .biasInit(nonZeroBias).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                        new int[] {2, 2}).name("maxpool2").build())
                .layer(4, new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1})
                        .name("cnn3").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(384)
                        .build())
                .layer(5, new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1})
                        .name("cnn4").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(256)
                        .biasInit(nonZeroBias).build())
                .layer(6, new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1})
                        .name("cnn5").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(256)
                        .biasInit(nonZeroBias).build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                        new int[] {7, 7}) // TODO: fix input and put stride back to 2,2
                        .name("maxpool3").build())
                .layer(8, new DenseLayer.Builder().name("ffn1").nIn(256).nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005)).biasInit(nonZeroBias).dropOut(dropOut)
                        .build())
                .layer(9, new DenseLayer.Builder().name("ffn2").nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005)).biasInit(nonZeroBias).dropOut(dropOut)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output").nOut(outputNum).activation(Activation.SOFTMAX).build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0])).build();

        return new MultiLayerNetwork(conf);

    }
    
	/*
	 * VGG-16, from Very Deep Convolutional Networks for Large-Scale Image Recognition
	 *
	 * Deep Face Recognition
     * http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf
	 *
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
	 * Source:
	 * https://arxiv.org/abs/1409.1556
	 * @author Justin Long (crockpotveggies)
	 *
	 * */
    public static MultiLayerNetwork VGG16(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum){

        int[] inputShape = new int[] {channels, height, width};
        WorkspaceMode workspaceMode;
        workspaceMode = WorkspaceMode.SEPARATE;
        ConvolutionLayer.AlgoMode cudnnAlgoMode;
        cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;


        MultiLayerConfiguration conf =
                new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(Updater.NESTEROVS).activation(Activation.RELU)
                        .trainingWorkspaceMode(workspaceMode).inferenceWorkspaceMode(workspaceMode)
                        .list()
                        // block 1
                        .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                .cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(64).cudnnAlgoMode(
                                        cudnnAlgoMode)
                                .build())
                        .layer(2, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build())
                        // block 2
                        .layer(3, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(5, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build())
                        // block 3
                        .layer(6, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(8, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(9, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build())
                        // block 4
                        .layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(13, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build())
                        // block 5
                        .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                        .layer(17, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build())
                        //                .layer(18, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                        //                        .build())
                        //                .layer(19, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                        //                        .build())
                        .layer(18, new OutputLayer.Builder(
                                LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                                .nOut(outputNum).activation(Activation.SOFTMAX) // radial basis function required
                                .build())
                        .backprop(true).pretrain(false).setInputType(InputType
                        .convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                        .build();

        return new MultiLayerNetwork(conf);
    }
    
	/*
	 *  A simple convolutional network for generic image classification.
	 *
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
	 * Source:
	 * https://github.com/oarriaga/face_classification/
	 * @author Justin Long (crockpotveggies)
	 *
	 * */
    public static MultiLayerNetwork simpleCNN(long seed, int iterations, double learningRate, int height, int width, int channels, int outputNum){

        int[] inputShape = new int[] {channels, height, width};
        WorkspaceMode workspaceMode;
        workspaceMode = WorkspaceMode.SEPARATE;

        MultiLayerConfiguration conf =
                new NeuralNetConfiguration.Builder().trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode).seed(seed).iterations(iterations)
                        .activation(Activation.IDENTITY).weightInit(WeightInit.RELU)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new AdaDelta()).regularization(false)
                        .convolutionMode(ConvolutionMode.Same).list()
                        // block 1
                        .layer(0, new ConvolutionLayer.Builder(new int[] {7, 7}).name("image_array")
                                .nIn(inputShape[0]).nOut(16).build())
                        .layer(1, new BatchNormalization.Builder().build())
                        .layer(2, new ConvolutionLayer.Builder(new int[] {7, 7}).nIn(16).nOut(16)
                                .build())
                        .layer(3, new BatchNormalization.Builder().build())
                        .layer(4, new ActivationLayer.Builder().activation(Activation.RELU).build())
                        .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                                new int[] {2, 2}).build())
                        .layer(6, new DropoutLayer.Builder(0.5).build())

                        // block 2
                        .layer(7, new ConvolutionLayer.Builder(new int[] {5, 5}).nOut(32).build())
                        .layer(8, new BatchNormalization.Builder().build())
                        .layer(9, new ConvolutionLayer.Builder(new int[] {5, 5}).nOut(32).build())
                        .layer(10, new BatchNormalization.Builder().build())
                        .layer(11, new ActivationLayer.Builder().activation(Activation.RELU).build())
                        .layer(12, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                                new int[] {2, 2}).build())
                        .layer(13, new DropoutLayer.Builder(0.5).build())

                        // block 3
                        .layer(14, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(64).build())
                        .layer(15, new BatchNormalization.Builder().build())
                        .layer(16, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(64).build())
                        .layer(17, new BatchNormalization.Builder().build())
                        .layer(18, new ActivationLayer.Builder().activation(Activation.RELU).build())
                        .layer(19, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                                new int[] {2, 2}).build())
                        .layer(20, new DropoutLayer.Builder(0.5).build())

                        // block 4
                        .layer(21, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(128).build())
                        .layer(22, new BatchNormalization.Builder().build())
                        .layer(23, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(128).build())
                        .layer(24, new BatchNormalization.Builder().build())
                        .layer(25, new ActivationLayer.Builder().activation(Activation.RELU).build())
                        .layer(26, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                                new int[] {2, 2}).build())
                        .layer(27, new DropoutLayer.Builder(0.5).build())


                        // block 5
                        /*.layer(28, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(256).build())
                        .layer(29, new BatchNormalization.Builder().build())
                        .layer(30, new ConvolutionLayer.Builder(new int[] {3, 3}).nOut(outputNum)
                                .build())
                        .layer(31, new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
                        .layer(32, new ActivationLayer.Builder().activation(Activation.SOFTMAX).build())*/
                        //trecho retirado de vgg16
                        .layer(28, new OutputLayer.Builder(
                                LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                                .nOut(outputNum).activation(Activation.SOFTMAX) // radial basis function required
                                .build())

                        .setInputType(InputType.convolutional(inputShape[2], inputShape[1],
                                inputShape[0]))
                        .backprop(true).pretrain(false).build();

        return new MultiLayerNetwork(conf);
    }



}
