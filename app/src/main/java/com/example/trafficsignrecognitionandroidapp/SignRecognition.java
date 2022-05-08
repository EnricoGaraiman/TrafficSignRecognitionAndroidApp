package com.example.trafficsignrecognitionandroidapp;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
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

public class SignRecognition {
    private String TAG = "ObjectDetection";

    // interpreters
    private Interpreter detectionInterpreter;
    private Interpreter recognitionInterpreter;

    // data members
    private List<String> labelList;
    private int pixelSize = 3; // rgb
    private final boolean quantized = false;
    private int threadsDetection = 2;
    private int threadsRecognition = 1;
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
    SignRecognition(AssetManager assetManager) throws IOException {
        // define GPU/CPU and number of threads
        Interpreter.Options optionsDetection = new Interpreter.Options();
        optionsDetection.setNumThreads(threadsDetection);

        // load detection model
        detectionInterpreter = new Interpreter(loadModelFile(assetManager, detectionPathModel), optionsDetection);

        // define GPU/CPU and number of threads
        Interpreter.Options optionsRecognition = new Interpreter.Options();
        optionsRecognition.setNumThreads(threadsRecognition);

        // load recognition model
        recognitionInterpreter = new Interpreter(loadModelFile(assetManager, recognitionPathModel), optionsRecognition);

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
    public void drawBoxes(Map<Integer, Object> recognitionOutputMap, Mat matImg, List<String> listOfResults, List<Integer> displayedSignClass, boolean realTime, String FPS) {
        // check input data
        if(recognitionOutputMap.size() == 0 || matImg.empty()) {
            return;
        }

        // rotate image if real time images from camera
        Mat matImgRotate = matImg;

        // rotate matImg
        if (realTime) {
            Core.flip(matImg.t(), matImgRotate, 1);
        }

        // initializations
        int frameWidth = matImgRotate.width();
        int frameHeight = matImgRotate.height();
        float scoreValue;
        String displayedText, classText, accText;
        List<String> displayedTextArray = new ArrayList<>();

        // detection latency
        int latency = 0;
        if(recognitionOutputMap.size() != 0) {
            latency = (int) recognitionOutputMap.get(0);
        }

        // for each recognition, draw result
        for(int i = 1; i < recognitionOutputMap.size(); i ++) {
            float[] res = (float[]) recognitionOutputMap.get(i);
            scoreValue = res[0];
            latency += res[7];

            if (scoreValue > confidence) {
                // draw rectangle in Original frame
                Imgproc.rectangle(matImgRotate,
                        new Point((res[1] * frameWidth - res[3] * frameWidth / 2), (res[2] * frameHeight - res[4] * frameHeight / 2)),
                        new Point((res[1] * frameWidth + res[3] * frameWidth / 2), (res[2] * frameHeight + res[4] * frameHeight / 2)),
                        new Scalar(250, 153, 28, 255), 2);

                // set class and accuracy as text
                classText = labelList.get((int) res[6]);
                accText = "(" + String.format("%.2f", res[5] * 100) + "%)";
                displayedText =  classText + " " + accText;
                displayedTextArray.add(displayedText);
                displayedSignClass.add(0, (int) res[6]);

                // write text on frame
                Imgproc.putText(matImgRotate,
                        classText,
                        new Point(res[1] * frameWidth - res[3] * frameWidth / 2, res[2] * frameHeight - res[4] * frameHeight / 2 - 26),
                        1, 1, new Scalar(28, 118, 143, 255), 2);

                Imgproc.putText(matImgRotate,
                        accText,
                        new Point(res[1] * frameWidth - res[3] * frameWidth / 2, res[2] * frameHeight - res[4] * frameHeight / 2 - 6),
                        1, 1, new Scalar(28, 118, 143, 255), 2);
                }
            }

        // rotate image if real time images from camera
        if (realTime) {
            Core.flip(matImgRotate.t(), matImg, 0);
        }

        // get list of results
        getListOfResults(listOfResults, displayedTextArray, latency, realTime, FPS);

        // log total latency
        Log.d(TAG, "drawBoxes: Total latency: " + latency + " ms");
    }

    private Map<Integer, Object> recognition(Map<Integer, Object> outputMap, Mat detectedImg, int latencyStorage) {
        Map<Integer, Object> recognitionOutputMap = new HashMap<>();

        // check input data
        if(detectedImg == null || detectedImg.empty() || outputMap.size() == 0) {
            return recognitionOutputMap;
        }

        // resize detectedMatImgRotate
        Mat resizedDetectedMatImgRotate = new Mat(detectionModelInputSize, detectionModelInputSize, CvType.CV_8UC4);
        Imgproc.resize(detectedImg, resizedDetectedMatImgRotate, new Size(detectionModelInputSize, detectionModelInputSize), Imgproc.INTER_AREA);

        // initializations
        int detectedWidth = resizedDetectedMatImgRotate.width();
        int detectedHeight = resizedDetectedMatImgRotate.height();
        float scoreValue;
        float[] recognition;
        int detectionLatency;
        List<float[]> showedResults = new ArrayList<>();
        int croppedX, croppedY, croppedW, croppedH;

        // get detection latency
        if (latencyStorage != 0) {
            detectionLatency = latencyStorage;
        }
        else {
            detectionLatency = (int) Objects.requireNonNull(outputMap.get(1));
        }
        recognitionOutputMap.put(0, detectionLatency);

        // get first N results
        float[][] result = getFirstNResults((float[][]) Array.get(Objects.requireNonNull(outputMap.get(0)), 0));

        // for each detection box, make recognition
        int i = 1;
        for (float[] res : result) {
            float[] recognitionResult = new float[8];
            scoreValue = res[0];
            if (scoreValue > confidence) {

                // check if a detection overlay showed detections
                if (!checkOverlayedBoxes(showedResults, res, detectedWidth, detectedHeight)) {

                    // get cropped image coordinates
                    croppedX = (int) (res[1] * detectedWidth - (res[3] * detectedWidth) / 2);
                    croppedY = (int) (res[2] * detectedHeight - (res[4] * detectedHeight) / 2);
                    croppedW = (int) (res[3] * detectedWidth);
                    croppedH = (int) (res[4] * detectedHeight);

                    if(croppedX > 0 && croppedY > 0 && croppedW > 0 && croppedH > 0 && !resizedDetectedMatImgRotate.empty()) {
                        // crop image
                        Rect rect = new Rect(croppedX, croppedY, croppedW, croppedH);
                        Mat croppedImg = resizedDetectedMatImgRotate.submat(rect);

//                        // save in storage
//                        Bitmap bitmap = Bitmap.createBitmap(croppedImg.cols(), croppedImg.rows(), Bitmap.Config.ARGB_8888);
//                        Utils.matToBitmap(croppedImg, bitmap);
////                        Bitmap bitmapDetected = Bitmap.createBitmap(resizedDetectedMatImgRotate.cols(), resizedDetectedMatImgRotate.rows(), Bitmap.Config.ARGB_8888);
////                        Utils.matToBitmap(resizedDetectedMatImgRotate, bitmapDetected);
//                        try {
//                            FileOutputStream out = new FileOutputStream(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + "/test" + ((int) (Math.random()*(10 - 1))) + 1 + ".png");
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

                            // add this result to showed results (remove overlayed detections)
                            showedResults.add(res);

                            // recognitionResult
                            recognitionResult[0] = res[0]; // detection accuracy
                            recognitionResult[1] = res[1]; // x
                            recognitionResult[2] = res[2]; // y
                            recognitionResult[3] = res[3]; // w
                            recognitionResult[4] = res[4]; // h
                            recognitionResult[5] = recognition[0]; // class
                            recognitionResult[6] = recognition[1]; // recognition accuracy
                            recognitionResult[7] = recognition[2]; // recognition latency
                            recognitionOutputMap.put(i, recognitionResult);
                            i++;
                        }
                    }
                }
            }
        }

        // return results
        return recognitionOutputMap;
    }

    /*------------------------------*/
    /* Traffic sign recognition     */
    /*------------------------------*/
    private float[] recognitionImage(Mat matImg) {
        float[] result = new float[3];

        // convert to bitmap
        Bitmap bitmap = Bitmap.createBitmap(matImg.cols(), matImg.rows(), Bitmap.Config.ARGB_8888);
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

        // measure latency
        int startTime = (int) System.currentTimeMillis();

        // make recognition
        recognitionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latency
        int stopTime = (int) System.currentTimeMillis();

        // get results
        float[] accAndClass = getAccuracyAndClassRecognition((float[]) Array.get(Objects.requireNonNull(outputMap.get(0)), 0));
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
        // output
        Map<Integer, Object> outputMap = new HashMap<>();

        // check if empty image
        if(matImg == null || matImg.empty()) {
            return recognition(outputMap, matImg, 0);
        }

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
        outputMap.put(0, new float[1][25200][6]);

        // measure latency
        int startTime = (int) System.currentTimeMillis();

        // prediction
        detectionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latency
        int stopTime = (int) System.currentTimeMillis();
        outputMap.put(1, stopTime - startTime);

        // return result for drawing
        return recognition(outputMap, matImgRotate, 0);
    }

    /*------------------------------*/
    /* Recognize photo from storage */
    /*------------------------------*/
    public Mat detectionImage(Mat matImg, List<String> listOfResults, List<Integer> displayedSignClass) {
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

        // measure latency
        int startTime = (int) System.currentTimeMillis();

        // detection
        detectionInterpreter.runForMultipleInputsOutputs(input, outputMap);

        // get latency
        int stopTime = (int) System.currentTimeMillis();

        // make recognition
        Map<Integer, Object> recognitionOutputMap = recognition(outputMap, matImg, stopTime - startTime);

        // draw boxes and return modified image
        drawBoxes(recognitionOutputMap, matImg, listOfResults, displayedSignClass, false, "0");
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
    private boolean checkOverlayedBoxes(List<float[]> showedResults, float[] result, int width, int height) {
        float xA, yA, xB, yB;
        float areaA, areaB, areaThreshold = 10;

        for (float[] res : showedResults) {
            xA = Math.max(result[1] * width - result[3] * width / 2, res[1] * width - res[3] * width / 2);
            yA = Math.max(result[2] * height - result[4] * height / 2, res[2] * height - res[4] * height / 2);
            xB = Math.min(result[1] * width + result[3] * width / 2, res[1] * width + res[3] * width / 2);
            yB = Math.min(result[2] * height + result[4] * height / 2, res[2] * height + res[4] * height / 2);

            // if boxes intersect
            areaA = Math.max(0, xB - xA + 1);
            areaB = Math.max(0, yB - yA + 1);
            if (areaA * areaB > 0 && (areaA > areaThreshold && areaB > areaThreshold)) {
                return true;
            }
        }

        return false;
    }

    /*----------------------------------------------*/
    /* Add recognition results in a list of results */
    /*----------------------------------------------*/
    private void getListOfResults(List<String> listOfResults, List<String> displayedTextArray, float latency, boolean realTime, String FPS) {
        String returnedText = "";

        if(realTime) {
            returnedText += displayedTextArray.size() + " signs";
            returnedText += "\n" + latency + " ms";
            returnedText += "\n" + FPS.split("@")[0];
            if(listOfResults.size() > 0) {
                listOfResults.remove(0);
            }
            listOfResults.add(returnedText);
        }
        else {
            returnedText += "Number of detected signs: " + displayedTextArray.size() + "\n\n";
            for (String text : displayedTextArray) {
                returnedText += text + "\n";
            }
            returnedText += "\nTotal latency: " + latency + " ms";
            listOfResults.add(0, returnedText);
        }
    }

}