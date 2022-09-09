package test;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.nd4j.linalg.dataset.DataSet;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
//import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

public class App {

	public static void main(String[] args) throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		
		int height=48;
		int width=48;
        int channels=1;// signe channel for graysacle image
        int outputNum=7;// 7 digits classification
        int batchSize=54;
        int epochCount=1;
        int seed =1234;
        Map<Integer,Double> learningRateByIterations=new HashMap<Integer, Double>();
        learningRateByIterations.put(0,0.06);
        learningRateByIterations.put(200,0.05);
        learningRateByIterations.put(600,0.028);
        learningRateByIterations.put(800,0.006);
        learningRateByIterations.put(1000,0.001);
        double quadraticError=0.0005;
        double momentum=0.9;
        Random randomGenNum=new Random(seed);
        
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(quadraticError)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION,learningRateByIterations),momentum))
                .weightInit(WeightInit.XAVIER)
                .list()
                    .layer(0,new ConvolutionLayer.Builder()
                            .kernelSize(3,3)
                            .nIn(channels)
                            .stride(1,1)
                            .nOut(20)
                            .activation(Activation.RELU).build())
                     .layer(1, new SubsamplingLayer.Builder()
                             .poolingType(SubsamplingLayer.PoolingType.MAX)
                             .kernelSize(2,2)
                             .stride(2,2)
                             .build())
                    .layer(2, new ConvolutionLayer.Builder(3,3)
                            .stride(1,1)
                            .nOut(50)
                            .activation(Activation.RELU)
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2,2)
                            .stride(2,2)
                            .build())
                    .layer(4, new DenseLayer.Builder()
                            .activation(Activation.RELU)
                            .nOut(500)
                            .build())
                    .layer(5,new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .nOut(outputNum)
                            .build())
                    .setInputType(InputType.convolutionalFlat(height,width,channels))
                .backpropType(BackpropType.Standard)
                .build();
        //System.out.println(configuration.toJson());
        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        model.init();
        
        
        System.out.println("Entrainement du model ....");
        
        String path ="F:\\Studies\\Java\\archive";
        File fileTrain = new File(path+"\\train");
        FileSplit fileSpliteTrain =new FileSplit (fileTrain,NativeImageLoader.ALLOWED_FORMATS,randomGenNum);
        ParentPathLabelGenerator labelMarker=new ParentPathLabelGenerator();
        RecordReader recordReaderTRAIN = new ImageRecordReader(height,width,channels,labelMarker);
        recordReaderTRAIN.initialize(fileSpliteTrain);
        int labelIndex=1;
        DataSetIterator trainDataSetIterator=new RecordReaderDataSetIterator(recordReaderTRAIN,batchSize,labelIndex,outputNum);
        DataNormalization scaler=new ImagePreProcessingScaler(0,1);
        scaler.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(scaler);
        
       /*
        while(trainDataSetIterator.hasNext()) {
        	DataSet dataSet = trainDataSetIterator.next();   
        	INDArray features = dataSet.getFeatures();
        	INDArray labels = dataSet.getLabels();
        	System.out.println(features.shapeInfoToString());
        	System.out.println(labels.shapeInfoToString());
        	System.out.println("--------------------------------------");
        }
        */
        
        /*
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage=new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
       
        */
        for (int i = 0; i < epochCount; i++) {
        	 model.fit(trainDataSetIterator); 
        	
        } 
        
        System.out.println("Evaluation du model ....");
        
        File fileTest = new File(path+"\\test");
        FileSplit fileSpliteTest =new FileSplit (fileTest,NativeImageLoader.ALLOWED_FORMATS,randomGenNum);
        RecordReader recordReaderTest= new ImageRecordReader(height,width,channels,labelMarker);
        recordReaderTest.initialize(fileSpliteTest);
        DataSetIterator testDataSetIterator=new RecordReaderDataSetIterator(recordReaderTest,batchSize,labelIndex,outputNum);
        DataNormalization scalerTest=new ImagePreProcessingScaler(0,1);
        scaler.fit(testDataSetIterator);
        testDataSetIterator.setPreProcessor(scalerTest);
        
        
        Evaluation evaluation = new Evaluation();
        
        while(testDataSetIterator.hasNext()) {
        	DataSet dataSet = testDataSetIterator.next();  
        	
        	INDArray features = dataSet.getFeatures();
        	INDArray labels = dataSet.getLabels();
        	
        	INDArray predicted =model.output(features);
        	evaluation.eval(predicted,labels);
        	 	
        }
        
        System.out.println(evaluation.stats());
        
        
        
        
        
        ModelSerializer.writeModel(model,new File(path+"/model.zip"),true);

	}

}
