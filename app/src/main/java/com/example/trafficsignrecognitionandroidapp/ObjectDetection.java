package com.example.trafficsignrecognitionandroidapp;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class ObjectDetection {
    private static String TAG = "ObjectDetection";

    private Interpreter interpreter;

    private static List<String> labelList;
    private int inputSize;
    private int pixelSize = 3; // rgb
//    private int IMAGE_MEAN = 0;
//    private float IMAGE_STD = 255.0f;
    private static final boolean quantized = false;
    private int threads = 1;
    private static float confidence = 0.5F;
    private int numberOfClasses;
//    private String pathModel = "nlcnn_model_99_64.tflite";
//    private String pathLabels = "labelmap.txt";
//    private int modelInputSize = 48;

    // use GPU in app
    private GpuDelegate gpuDelegate;
    private static int height = 0;
    private static int width = 0;

    ObjectDetection(AssetManager assetManager, String modelPath, String labelPath, int inputSize) throws IOException {
        this.inputSize = inputSize;

        // define GPU/CPU and number of threads
        Interpreter.Options options = new Interpreter.Options();
//        gpuDelegate = new GpuDelegate();
//        options.addDelegate(gpuDelegate);
        options.setNumThreads(threads);

        // load model
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);

        // load labels
        labelList = loadLabels(assetManager, labelPath);

        // number of classes
        numberOfClasses = labelList.size();
    }

    private List<String> loadLabels(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();

        // create reader and read line by line
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();

        // return label list
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // return description of file model
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long length = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }

    public static Mat drawBoxes(Map<Integer, Object> outputMap, Mat mat_img){
        // rotate image to get portrait image
        Mat mat_img_rotate = new Mat();
        Mat a = mat_img.t();
        Core.flip(a, mat_img_rotate, 1);
        a.release();

        Object value = outputMap.get(0);
        Object predict_class = outputMap.get(1);
        Object score = outputMap.get(2);

        for (int i = 0; i < 10; i++) {
            float class_value = (float) Array.get(Array.get(predict_class, 0), i);
            float score_value = (float) Array.get(Array.get(score, 0), i);
            // define threshold for score
            if (score_value > confidence) {
                Object box1 = Array.get(Array.get(value, 0), i);
                // multiplying it with original height and width of frame

                float top = (float) Array.get(box1, 0) * height;
                float left = (float) Array.get(box1, 1) * width;
                float bottom = (float) Array.get(box1, 2) * height;
                float right = (float) Array.get(box1, 3) * width;
                // draw rectangle in Original frame //  starting point    // ending point of box  // color of box      // thickness
                Imgproc.rectangle(mat_img_rotate, new Point(left, top), new Point(right, bottom), new Scalar(0, 255, 0, 255), 2);
                // write text on frame
                // string of class name of object  // starting point                         // color of text           // size of text
                Imgproc.putText(mat_img_rotate, labelList.get((int) class_value), new Point(left, top), 3, 1, new Scalar(255, 0, 0, 255), 2);
            }

        }

        // before return rotate back with 90 degree
        Mat b = mat_img_rotate.t();
        Core.flip(b, mat_img, 0);
        b.release();
        return mat_img;
    }

    public Map<Integer, Object> recognizeImage(Mat mat_img) {
        // measure latency
        long startTime = System.currentTimeMillis();

        // rotate image to get portrait image
        Mat mat_img_rotate = new Mat();
        Mat a = mat_img.t();
        Core.flip(a, mat_img_rotate, 1);
        a.release();

        // convert to bitmap
        Bitmap bitmap = null;
        bitmap = Bitmap.createBitmap(mat_img_rotate.cols(), mat_img_rotate.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_img_rotate, bitmap);

        // define h and w
        height = bitmap.getHeight();
        width = bitmap.getWidth();

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false);

        // convert bitmap to bytebuffer -> input
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // output
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][numberOfClasses]);

        // prediction
        interpreter.runForMultipleInputsOutputs(input, outputMap);

        Log.d(TAG, "recognizeImage: a iesit? ");
//        Log.d(TAG, "recognizeImage: " + getIndexOfLargest(outputMap.get(0))));
        Log.e(TAG, "recognizeImage: " + labelList.get(getIndexOfLargest((float[]) Array.get(outputMap.get(0), 0))));
//        Log.d(TAG, "recognizeImage: out? " + getIndexOfLargest(outputMap.get(0)));
//        try {
//            Thread.sleep(1000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        long stopTime = System.currentTimeMillis();
        Log.d(TAG, "Elapsed time was " + (stopTime - startTime) + " milliseconds.");

//        return outputMap;
        Map<Integer, Object> output_map = new TreeMap<>();

                float[][][] boxes = new float[1][10][4];// first 10 object detected + 4 coordinates
        float[][] scores = new float[1][10];// scores
        float[][] classes = new float[1][10];// classes

        // add to object map
        output_map.put(0, boxes);
        output_map.put(1, scores);
        output_map.put(2, classes);
        return output_map; // schimba
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;

        // model input
        if (quantized) {
            byteBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * pixelSize);
        } else {
            byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * pixelSize);
        }

        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int val = intValues[pixel++];
                if (quantized) {
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF)) / 255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF)) / 255.0f);
                    byteBuffer.putFloat((((val) & 0xFF)) / 255.0f);
                }
            }
        }

        return byteBuffer;
    }

    public int getIndexOfLargest( float[] array )
    {
        if ( array == null || array.length == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }

    public Mat recognizePhoto(Mat mat_img) {
        // measure delay
        long startTime = System.currentTimeMillis();

        // convert to bitmap
        Bitmap bitmap = null;
        bitmap = Bitmap.createBitmap(mat_img.cols(), mat_img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_img, bitmap);

        // define h and w
        height = bitmap.getHeight();
        width = bitmap.getWidth();

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false);

        // convert bitmap to bytebuffer as model input should be ??
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

        // define output - boxes, score, classes
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        Map<Integer, Object> outputMap = new TreeMap<>();

        float[][][] boxes = new float[1][10][4];// first 10 object detected + 4 coordinates
        float[][] scores = new float[1][10];// scores
        float[][] classes = new float[1][10];// scores

        // add to object map
        outputMap.put(1, boxes);
        outputMap.put(0, scores);
        outputMap.put(3, classes);

        // prediction
        interpreter.runForMultipleInputsOutputs(input, outputMap);

        // draw boxex
        Object value = outputMap.get(1);
        Object predict_class = outputMap.get(0);
        Object score = outputMap.get(3);

        for (int i = 0; i < 10; i++) {
            float class_value = (float) Array.get(Array.get(predict_class, 0), i);
            float score_value = (float) Array.get(Array.get(score, 0), i);
            // define threshold for score
            if (score_value > confidence) {
                Object box1 = Array.get(Array.get(value, 0), i);
                // multiplying it with original height and width of frame

                float top = (float) Array.get(box1, 0) * height;
                float left = (float) Array.get(box1, 1) * width;
                float bottom = (float) Array.get(box1, 2) * height;
                float right = (float) Array.get(box1, 3) * width;
                // draw rectangle in Original frame //  starting point    // ending point of box  // color of box      // thickness
                Imgproc.rectangle(mat_img, new Point(left, top), new Point(right, bottom), new Scalar(0, 255, 0, 255), 2);
                // write text on frame
                // string of class name of object  // starting point                         // color of text           // size of text
                Imgproc.putText(mat_img, labelList.get((int) class_value), new Point(left, top), 3, 1, new Scalar(255, 0, 0, 255), 2);
            }

        }

        long stopTime = System.currentTimeMillis();
        Log.d(TAG, "Elapsed time was " + (stopTime - startTime) + " milliseconds.");

        return mat_img;
    }

}
