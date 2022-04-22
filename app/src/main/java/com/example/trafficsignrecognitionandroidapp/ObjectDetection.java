package com.example.trafficsignrecognitionandroidapp;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

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
    private int pixelSize = 3; // rgb
    private final boolean quantized = false;
    private int threads = 4;
    private float confidence = 0.5F;
    private int numberOfClasses;
    private int numberOfDetection = 10;

    // detection & recognition
    private String detectionPathModel = "yolov5n.tflite";
    private String recognitionPathModel = "nlcnn_model_99_64.tflite";
    private String pathLabels = "labelmap.txt";
    private int detectionModelInputSize = 640;
    private int recognitionModelInputSize = 48;

    /*------------------------------*/
    /* ObjectDetection constructor  */
    /*------------------------------*/
    ObjectDetection(AssetManager assetManager) throws IOException {
        // define GPU/CPU and number of threads
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(threads);

        // load detection model
        detectionInterpreter = new Interpreter(loadModelFile(assetManager, detectionPathModel), options);

        // load recognition model
        recognitionInterpreter = new Interpreter(loadModelFile(assetManager, recognitionPathModel), options);

        // load labels
        labelList = loadLabels(assetManager, pathLabels);

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
    public void drawBoxes(Map<Integer, Object> outputMap, Mat matImg, Mat detectedImg, long latencyStorage, boolean realTime) {
        // rotate image if real time images from camera
        Mat matImgRotate = matImg;
        Mat detectedMatImgRotate = detectedImg;

        if (realTime) {
            // rotate matImg
            Mat a = matImg.t();
            Core.flip(a, matImgRotate, 1);
            a.release();

            // rotate detectedImg
            Mat b = detectedImg.t();
            Core.flip(b, detectedMatImgRotate, 1);
            b.release();
        }

        // resize detectedMatImgRotate
        Mat resizedDetectedMatImgRotate = new Mat();
        Imgproc.resize(detectedMatImgRotate, resizedDetectedMatImgRotate, new Size(detectionModelInputSize, detectionModelInputSize), Imgproc.INTER_AREA);

        // initializations
        int detectedWidth = resizedDetectedMatImgRotate.width();
        int detectedHeight = resizedDetectedMatImgRotate.height();
        int frameWidth = matImgRotate.width();
        int frameHeight = matImgRotate.height();
        float aspectRatio = realTime ? (float) frameHeight/frameWidth : 1;
        float scoreValue;
        float[] recognition;
        long latency;
        int padding = 10;
        List<float[]> showedResults = new ArrayList<>();

        // get detection latency
        if (latencyStorage != 0) {
            latency = latencyStorage;
        }
        else {
            latency = (long) Objects.requireNonNull(outputMap.get(1));
        }

        // get first N results and draw boxes
        float[][] result = getFirstNResults((float[][]) Array.get(Objects.requireNonNull(outputMap.get(0)), 0));

        for (float[] res : result) {
            scoreValue = res[0];
            if (scoreValue > confidence) {

                // check if a detection overlay showed detections
                if (!checkOverlayedDetections(showedResults, res, detectedWidth, detectedHeight)) {
                    try {
                        // crop image
                        Rect rect = new Rect(
                                (int) (res[1] * detectedWidth - padding / 2 - (res[3] * detectedWidth + padding) / 2),
                                (int) (res[2] * detectedHeight - padding / 2 - (res[4] * detectedHeight + padding) / 2),
                                (int) (res[3] * detectedWidth + padding),
                                (int) (res[4] * detectedHeight + padding)
                        );
                        Mat croppedImg = resizedDetectedMatImgRotate.submat(rect);

//                        // save in storage
//                        Bitmap bitmap = Bitmap.createBitmap(croppedImg.cols(), croppedImg.rows(), Bitmap.Config.ARGB_8888);
//                        Utils.matToBitmap(croppedImg, bitmap);
////                        Bitmap bitmapDetected = Bitmap.createBitmap(resizedDetectedMatImgRotate.cols(), resizedDetectedMatImgRotate.rows(), Bitmap.Config.ARGB_8888);
////                        Utils.matToBitmap(resizedDetectedMatImgRotate, bitmapDetected);
//                        try {
//                            FileOutputStream out = new FileOutputStream(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + "/test" + i + ".png");
//                            bitmap.compress(Bitmap.CompressFormat.PNG, 90, out);
//                            out.close();
//
////                            FileOutputStream out1 = new FileOutputStream(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + "/test-detected.png");
////                            bitmapDetected.compress(Bitmap.CompressFormat.PNG, 90, out1);
////                            out1.close();
//                        }
//                        catch (Exception e) {
//                            e.printStackTrace();
//                        }

                        if (croppedImg.rows() > 0 && croppedImg.cols() > 0) {
                            // make recognition traffic sign
                            recognition = recognitionImage(croppedImg);

                            // draw rectangle in Original frame
                            Imgproc.rectangle(matImgRotate,
                                    new Point((res[1] * frameWidth - res[3] * frameWidth * aspectRatio / 2), (res[2] * frameHeight - res[4] * frameHeight / aspectRatio / 2)),
                                    new Point((res[1] * frameWidth + res[3] * frameWidth * aspectRatio / 2), (res[2] * frameHeight + res[4] * frameHeight / aspectRatio / 2)),
                                    new Scalar(250, 153, 28, 255), 2);

                            // write text on frame
                            Imgproc.putText(matImgRotate,
                                    labelList.get((int) recognition[1]) + " (" + String.format("%.2f", recognition[0] * 100) + "%)",
                                    new Point(res[1] * frameWidth - res[3] * frameWidth * aspectRatio / 2, res[2] * frameHeight - res[4] * frameHeight / aspectRatio / 2 - 6),
                                    1, 1, new Scalar(251, 243, 242, 255), 2);

                            // add latency
                            latency += recognition[2];

                            // add this result to showed results (remove overlayed detections)
                            showedResults.add(res);
                        }
                    }
                    catch (Exception e) {
                        Log.e(TAG, "drawBoxes: Error: " + e.getMessage());
                    }
                }
            }
        }

        // rotate image if real time images from camera
        if (realTime) {
            // rotate matImg back
            Mat c = matImgRotate.t();
            Core.flip(c, matImg, 0);
            c.release();

            // rotate detectedImg back
            Mat d = detectedMatImgRotate.t();
            Core.flip(d, detectedImg, 0);
            d.release();
        }

        Log.e(TAG, "drawBoxes: Total latency: " + latency + "ms");
    }

    /*------------------------------*/
    /* Traffic sign recognition     */
    /*------------------------------*/
    private float[] recognitionImage(Mat matImg) {
        float[] result = new float[3];

        // measure latency
        long startTime = System.currentTimeMillis();

        // convert to bitmap
        Bitmap bitmap;
        bitmap = Bitmap.createBitmap(matImg.cols(), matImg.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matImg, bitmap);

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, recognitionModelInputSize, recognitionModelInputSize, false);

        // convert bitmap to bytebuffer -> input
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap, recognitionModelInputSize);
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // define output
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][numberOfClasses]);

        // make recognition
        recognitionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latency
        long stopTime = System.currentTimeMillis();

        // get results
        float[] accAndClass = getAccuracyAndClassRecognition((float[]) Array.get(outputMap.get(0), 0));
        result[0] = accAndClass[0]; // set accuracy
        result[1] = accAndClass[1]; // set class
        result[2] = stopTime - startTime;

        // return results
        return result;
    }

    /*-----------------------------*/
    /* Frame processing real time  */
    /*-----------------------------*/
    public Map<Integer, Object> detectionFrame(Mat matImg) {
        // measure latency
        long startTime = System.currentTimeMillis();

        // rotate image
        Mat matImgRotate = new Mat();
        Mat a = matImg.t();
        Core.flip(a, matImgRotate, 1);
        a.release();

        // convert to bitmap
        Bitmap bitmap;
        bitmap = Bitmap.createBitmap(matImgRotate.cols(), matImgRotate.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matImgRotate, bitmap);

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, detectionModelInputSize, detectionModelInputSize, false);

        // convert bitmap to bytebuffer -> input
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap, detectionModelInputSize);
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // output
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][25200][6]);

        // prediction
        detectionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latency
        long stopTime = System.currentTimeMillis();
        outputMap.put(1, stopTime - startTime);

        // return result for drawing
        return outputMap;
    }

    /*------------------------------*/
    /* Recognize photo from storage */
    /*------------------------------*/
    public Mat detectionImage(Mat matImg) {
        // measure latency
        long startTime = System.currentTimeMillis();

        // convert to bitmap
        Bitmap bitmap;
        bitmap = Bitmap.createBitmap(matImg.cols(), matImg.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matImg, bitmap);

        // scale to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, detectionModelInputSize, detectionModelInputSize, false);

        // convert bitmap to bytebuffer -> input
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap, detectionModelInputSize);
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // define output
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][25200][6]);

        // detection
        detectionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latence
        long stopTime = System.currentTimeMillis();

        // draw boxes and return modified image
        drawBoxes(outputMap, matImg, matImg, stopTime - startTime, false);
        return matImg;
    }

    /*-----------------------------------------------------*/
    /* Convert bitmap to bytes for input neuronal network  */
    /*-----------------------------------------------------*/
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, int inputSize) {
        ByteBuffer byteBuffer;

        // model input
        if (quantized) {
            byteBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * pixelSize);
        }
        else {
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
    private float[][] getFirstNResults(float[][] detection) {
        int index;
        float[][] result = new float[numberOfDetection][5];
        float[] probMax = new float[numberOfDetection];

        for (int i = 0; i < numberOfDetection; i++) {
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

            if (index != -1) {
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
    private boolean check(float[] arr, float toCheckValue) {
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
    private float[] getAccuracyAndClassRecognition(float[] array) {
        float[] result = new float[2];
        if (array == null || array.length == 0) return result; // null or empty

        int predClass = 0;
        float prob = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[predClass]) {
                predClass = i;
                prob = array[i];
            }
        }

        result[0] = prob;
        result[1] = predClass;
        return result; // position of the first largest found
    }

    /*-------------------------------*/
    /* Check overlayed detections    */
    /*-------------------------------*/
    private boolean checkOverlayedDetections(List<float[]> showedResults, float[] result, int width, int height) {
        float xA, yA, xB, yB;

        for (float[] res : showedResults) {
            xA = Math.max(result[1] * width - result[3] * width / 2, res[1] * width - res[3] * width / 2);
            yA = Math.max(result[2] * height - result[4] * height / 2, res[2] * height - res[4] * height / 2);
            xB = Math.min(result[1] * width + result[3] * width / 2, res[1] * width + res[3] * width / 2);
            yB = Math.min(result[2] * height + result[4] * height / 2, res[2] * height + res[4] * height / 2);

            // if boxes intersect
            if (Math.max(0, xB - xA + 1) * Math.max(0, yB - yA + 1) > 0) {
                return true;
            }
        }

        return false;
    }
}