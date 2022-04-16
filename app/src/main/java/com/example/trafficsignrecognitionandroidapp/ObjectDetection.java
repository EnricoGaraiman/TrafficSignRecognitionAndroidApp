package com.example.trafficsignrecognitionandroidapp;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
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
import java.util.Objects;

public class ObjectDetection {
    private String TAG = "ObjectDetection";

    private Interpreter detectionInterpreter;
    private Interpreter recognitionInterpreter;

    private List<String> labelList;
    private int detectionInputSize;
    private int recognitionInputSize;
    private int pixelSize = 3; // rgb
    private final boolean quantized = false;
    private int threads = 4;
    private float confidence = 0.5F;
    private int numberOfClasses;

    // use GPU in app
    private GpuDelegate gpuDelegate;
    private int originalHeight = 0;
    private int originalWidth = 0;
    private int numberOfDetection = 10;

    /*------------------------------*/
    /* ObjectDetection constructor  */
    /*------------------------------*/
    ObjectDetection(
            AssetManager assetManager,
            String detectionModelPath,
            String recognitionModelPath,
            String labelPath,
            int detectionInputSize,
            int recognitionInputSize
    ) throws IOException {
        this.detectionInputSize = detectionInputSize;
        this.recognitionInputSize = recognitionInputSize;

        // define GPU/CPU and number of threads
        Interpreter.Options options = new Interpreter.Options();
//        gpuDelegate = new GpuDelegate();
//        options.addDelegate(gpuDelegate);
        options.setNumThreads(threads);

        // load detection model
        detectionInterpreter = new Interpreter(loadModelFile(assetManager, detectionModelPath), options);

        // load recognition model
        recognitionInterpreter = new Interpreter(loadModelFile(assetManager, recognitionModelPath), options);

        // load labels
        labelList = loadLabels(assetManager, labelPath);

        // number of classes
        numberOfClasses = labelList.size();
    }

    /*------------------------------*/
    /* Load labels for recognition  */
    /*------------------------------*/
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

    /*------------------------------*/
    /* Load TFLite model            */
    /*------------------------------*/
    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // return description of file model
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long length = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }

    /*------------------------------*/
    /* Draw boxes after detection   */
    /*------------------------------*/
    public void drawBoxes(Map<Integer, Object> outputMap, Mat mat_img, Mat detected_img){
        // get first N results
        int width = originalWidth;
        int height = originalHeight;
        float[][] result = getFirstNResults((float[][])Array.get(Objects.requireNonNull(outputMap.get(0)), 0));
        float x, y, w, h;
        float [] recognition;

        // get first N results and draw boxes
        for (float[] res : result) {
            float score_value = res[0];
            if (score_value > confidence) {
                // multiplying it with original height and width of frame
                x = res[1] * width;
                y = res[2] * height;
                w = res[3] * width;
                h = res[4] * height;

                // crop image
                Rect rect = new Rect((int)(x - w / 2), (int)(y - h / 2) , (int)w, (int)h);
                Mat croppedImg = detected_img.submat(rect);//new Mat(detected_img, rect);
                Log.e(TAG, "drawBoxes: " + croppedImg.rows() + ' ' + croppedImg.cols() );
                if(croppedImg.rows() > 0 && croppedImg.cols() > 0) {
                    // make recognition traffic sign
                    recognition = recognitionImage(croppedImg);

                    // draw rectangle in Original frame
                    Imgproc.rectangle(mat_img, new Point(x - w / 2, y - h / 2),
                            new Point(x + w / 2, y + h / 2),
                            new Scalar(0, 255, 0, 255), 2);
                    // write text on frame
                    Imgproc.putText(mat_img,
                            labelList.get((int) recognition[1]) + "(" + String.format("%.2f", recognition[0] * 100) + "%)",
                            new Point(x - w / 2, y - h / 2), 2, 1, new Scalar(255, 0, 0, 255), 2);
                }
            }
        }
    }

    /*------------------------------*/
    /* Traffic sign recognition     */
    /*------------------------------*/
    private float[] recognitionImage(Mat mat_img) {
        float []result = new float[3];

        // measure latency
        long startTime = System.currentTimeMillis();

        // convert to bitmap
        Bitmap bitmap;
        bitmap = Bitmap.createBitmap(mat_img.cols(), mat_img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_img, bitmap);

        // define h and w
        originalHeight = bitmap.getHeight();
        originalWidth = bitmap.getWidth();

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, recognitionInputSize, recognitionInputSize, false);

        // convert bitmap to bytebuffer -> input
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap, recognitionInputSize);
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // define output
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][numberOfClasses]);

        // make recognition
        recognitionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latency
        long stopTime = System.currentTimeMillis();
        long latency = stopTime - startTime;
        Log.d(TAG, "Elapsed time was " + latency + " milliseconds.");

        // get results
        float []accAndClass = getAccuracyAndClassRecognition((float[]) Array.get(outputMap.get(0), 0));
        result[0] = accAndClass[0]; // set accuracy
        result[1] = accAndClass[1]; // set class
        result[2] = latency; // set latency

        // return results
        return result;
    }

    /*-----------------------------*/
    /* Frame processing real time  */
    /*-----------------------------*/
    public Map<Integer, Object> recognizeFrame(Mat mat_img) {
        // measure latency
        long startTime = System.currentTimeMillis();

        // convert to bitmap
        Bitmap bitmap;
        bitmap = Bitmap.createBitmap(mat_img.cols(), mat_img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_img, bitmap);

        // define h and w
        originalHeight = bitmap.getHeight();
        originalWidth = bitmap.getWidth();

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, detectionInputSize, detectionInputSize, false);

        // convert bitmap to bytebuffer -> input
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap, detectionInputSize);
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // output
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][25200][6]);

        // prediction
        detectionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        Log.d(TAG, "recognizeImage: a iesit? ");
//        try {
//            Thread.sleep(1000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        long stopTime = System.currentTimeMillis();
        long latency = stopTime - startTime;
        Log.d(TAG, "Elapsed time was " + latency + " milliseconds.");

        // return result for drawing
        return outputMap;
    }

    /*------------------------------*/
    /* Recognize photo from storage */
    /*------------------------------*/
    public Mat recognizePhoto(Mat mat_img) {
        // measure latency
        long startTime = System.currentTimeMillis();

        // convert to bitmap
        Bitmap bitmap;
        bitmap = Bitmap.createBitmap(mat_img.cols(), mat_img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_img, bitmap);

        // define h and w of image
        originalHeight = bitmap.getHeight();
        originalWidth = bitmap.getWidth();

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, detectionInputSize, detectionInputSize, false);

        // convert bitmap to bytebuffer -> input
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap, detectionInputSize);
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // define output
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][25200][6]);

        // detection
        detectionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latence
        long stopTime = System.currentTimeMillis();
        long latency = stopTime - startTime;
        Log.d(TAG, "Elapsed time was " + latency + " milliseconds.");

        // draw boxes and return modified image
        drawBoxes(outputMap, mat_img, mat_img);
        return mat_img;
    }

    /*-----------------------------------------------------*/
    /* Convert bitmap to bytes for input neuronal network  */
    /*-----------------------------------------------------*/
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, int inputSize) {
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

    /*------------------------*/
    /* Get first N detection  */
    /*------------------------*/
    public float[][] getFirstNResults(float[][] detection) {
        int index;
        float[][] result = new float[numberOfDetection][5];
        float[] probMax = new float [numberOfDetection];

        for (int i = 0; i < numberOfDetection; i ++) {
            index = -1;
            for (int j = 0; j < detection.length; j++) {
                if (
                        probMax[i] < detection[j][4] &&
                        !check(probMax, detection[j][4]) &&
                        detection[j][4] >= confidence
                ) {
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

    /*----------------------------------*/
    /* Check if a value exist in array  */
    /*----------------------------------*/
    private boolean check(float[] arr, float toCheckValue)
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

    /*-------------------------------*/
    /* Get largest index from array  */
    /*-------------------------------*/
    public float[] getAccuracyAndClassRecognition(float[] array)
    {
        float [] result = new float[2];
        if ( array == null || array.length == 0 ) return result; // null or empty

        int predClass = 0;
        float prob = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[predClass] ) {
                predClass = i;
                prob = array[i];
            }
        }

        result[0] = prob;
        result[1] = predClass;
        return result; // position of the first largest found
    }
}
