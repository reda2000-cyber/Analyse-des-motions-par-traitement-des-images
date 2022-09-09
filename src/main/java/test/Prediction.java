package test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.swing.JFileChooser;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.PieChart;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;





public class Prediction  {
    private static final Logger logger = LoggerFactory.getLogger(App.class);
    private static final File modelPath = new File("F:\\Studies\\Java\\archive\\model.zip");
    private static final int height = 48;
    private static final int width = 48;
    private static final int channels = 1;

    
    public static String fileChose() {
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION) {
            File file = fc.getSelectedFile();
            return file.getAbsolutePath();
        } else {
            return null;
        }
    }
    

    public static void main(String[] args) throws Exception {
    	
    	
        if (!modelPath.exists()) {
            logger.info("Le modèle introuvable. L'avez-vous entraîné ?");
            return;
        }
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
        String path = fileChose();
        File file = new File(path);

        INDArray image = new NativeImageLoader(height, width, channels).asMatrix(file);
        new ImagePreProcessingScaler(0, 1).transform(image);

       
        INDArray output = model.output(image);
        logger.info("Fichier: {}", path);
        logger.info("Probabilities: {}", output);
        System.out.println(" [[     ANGRY,   DISGUST,      FEAR,     HAPPY,   NEUTRAL,       SAD,   SUPRISE]] ");
        System.out.println(" "+output);
        
        
        
		
		
        
        
       
          
    }

	

}