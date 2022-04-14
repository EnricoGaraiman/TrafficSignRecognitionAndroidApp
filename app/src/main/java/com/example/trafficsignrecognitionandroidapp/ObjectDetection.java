package com.example.trafficsignrecognitionandroidapp;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
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
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ObjectDetection {
    private static String TAG = "ObjectDetection";

    private Interpreter interpreter;

    private static List<String> labelList;
    private static int inputSize;
    private int pixelSize = 3; // rgb
    private static final boolean quantized = false;
    private int threads = 4;
    private static float confidence = 0.0001F;
    private int numberOfClasses;
//    private String pathModel = "nlcnn_model_99_64.tflite";
//    private String pathLabels = "labelmap.txt";
//    private int modelInputSize = 48;

    // use GPU in app
    private GpuDelegate gpuDelegate;
    private static int height = 0;
    private static int width = 0;
    private static int numberOfDetection = 10;

    ObjectDetection(AssetManager assetManager, String modelPath, String labelPath, int inputSize) throws IOException {
        this.inputSize = inputSize;

        // define GPU/CPU and number of threads
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
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

    public static void drawBoxes(Map<Integer, Object> outputMap, Mat mat_img){
        // rotate image to get portrait image

//        Object value = outputMap.get(0);
//        Object predict_class = outputMap.get(1);
//        Object score = outputMap.get(2);

        float[][] result = getFirstNResults((float[][])Array.get(outputMap.get(0), 0));

        // get first N results and draw boxes
        for (float[] res : result) {
            float score_value = res[0];
            if (score_value > confidence) {
                // multiplying it with original height and width of frame
                float x = res[1] * width;
                float y = res[2] * height;
                float w = res[3] * width;
                float h = res[4] * height;

                // draw rectangle in Original frame //  starting point    // ending point of box  // color of box      // thickness
                Imgproc.rectangle(mat_img, new Point(x - w / 2, y - h / 2), new Point(x + w / 2, y + h / 2), new Scalar(0, 255, 0, 255), 2);
                // write text on frame
                // string of class name of object  // starting point                         // color of text           // size of text
                Imgproc.putText(mat_img, "Sign", new Point(x, y), 3, 1, new Scalar(255, 0, 0, 255), 2);
            }
        }

    }

    public Map<Integer, Object> recognizeFrame(Mat mat_img) {
        // measure latency
        long startTime = System.currentTimeMillis();

        // rotate image to get portrait image
        Mat mat_img_rotate = new Mat();
        Mat a = mat_img.t();
        Core.flip(a, mat_img_rotate, 1);
        a.release();

        // convert to bitmap
        Bitmap bitmap = null;
        bitmap = Bitmap.createBitmap(mat_img.cols(), mat_img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_img, bitmap);

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
        //outputMap.put(0, new float[1][numberOfClasses]);
        outputMap.put(0, new float[1][25200][6]);

        // prediction
        interpreter.runForMultipleInputsOutputs(input, outputMap);

        Log.d(TAG, "recognizeImage: a iesit? ");
//        Log.d(TAG, "recognizeImage: " + getIndexOfLargest(outputMap.get(0))));
        //Log.e(TAG, "recognizeImage: " + labelList.get(getIndexOfLargest((float[]) Array.get(outputMap.get(0), 0)))); !!!!!!!!

        // yolov5s
//        List<?> results = Arrays.stream((float[][])Array.get(outputMap.get(0), 0)).filter( i -> i != null && i[0] != 0).collect(Collectors.toList());

//        float [][]result = parseDetectionOutput(outputMap);


//        Log.d(TAG, "recognizeImage: " + getIndexOfLargest((float[]) Array.get(Array.get(outputMap.get(0), 0), 0)));
//        Log.d(TAG, "recognizeImage: " + ((float[]) Array.get(Array.get(outputMap.get(0), 0),0)).length); //6
//        Log.d(TAG, "recognizeImage: " + ((float[][]) Array.get(outputMap.get(0), 0)).length); //25200

//        Log.d(TAG, "recognizeImage: out? " + getIndexOfLargest(outputMap.get(0)));
//        try {
//            Thread.sleep(1000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        long stopTime = System.currentTimeMillis();
        Log.d(TAG, "Elapsed time was " + (stopTime - startTime) + " milliseconds.");

//        return outputMap;
//        Map<Integer, Object> output_map = new TreeMap<>();
//
//        float[][][] boxes = new float[1][10][4];// first 10 object detected + 4 coordinates
//        float[][] scores = new float[1][10];// scores
//        float[][] classes = new float[1][10];// classes
//
//        // add to object map
//        output_map.put(0, boxes);
//        output_map.put(1, scores);
//        output_map.put(2, classes);
        return outputMap;
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

    public static float[][] getFirstNResults(float[][] detection) {
        int index;
        float[][] result = new float[numberOfDetection][5];
        float[] probMax = new float [numberOfDetection];

        for (int i = 0; i < numberOfDetection; i ++) {
            index = -1;
            for (int j = 0; j < detection.length; j++) {
                if (probMax[i] < detection[j][4] && !check(probMax, detection[j][4]) && detection[j][4] >= confidence) {
                    index = j;
                    probMax[i] = detection[j][4];
                }
            }

            if(index != -1) {
                result[i][0] = probMax[i];
                result[i][1] = detection[index][0];
                result[i][2] = detection[index][1];
                result[i][3] = detection[index][2];
                result[i][4] = detection[index][3];
            }

        }

        return result;
    }

    private static boolean check(float[] arr, float toCheckValue)
    {
        // check if the specified element
        // is present in the array or not
        // using Linear Search method
        boolean test = false;
        for (float element : arr) {
            if (element == toCheckValue) {
                test = true;
                break;
            }
        }
        return test;
    }

    public float[] getIndexOfLargest( float[] array )
    {
        float [] result = new float[2];
        if ( array == null || array.length == 0 ) return result; // null or empty

        int largest = 0;
        float prob = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) {
                largest = i;
                prob = array[i];
            }
        }

        result[0] = largest;
        result[1] = prob;
        return result; // position of the first largest found
    }

    public Mat recognizePhoto(Mat mat_img) {
        // measure latency
        long startTime = System.currentTimeMillis();

        // convert to bitmap
        Bitmap bitmap = null;
        bitmap = Bitmap.createBitmap(mat_img.cols(), mat_img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_img, bitmap);

        // define h and w of image
        height = bitmap.getHeight();
        width = bitmap.getWidth();

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false);

        // convert bitmap to bytebuffer -> input
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // define output
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][25200][6]);

        // prediction
        interpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latence
        long stopTime = System.currentTimeMillis();
        Log.d(TAG, "Elapsed time was " + (stopTime - startTime) + " milliseconds.");

        // draw boxes and return modified image
        drawBoxes(outputMap, mat_img);
        return mat_img;
    }

}
