package com.example.trafficsignrecognitionandroidapp;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.Handler;
import android.os.Looper;
import android.util.AttributeSet;
import android.util.Log;

import org.opencv.BuildConfig;
import org.opencv.android.JavaCameraView;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class JavaCameraDetectionView extends JavaCameraView {

    private static final String TAG = "CameraBridgeDetectionView";
    private CompletableFuture<?> detectionOutput;
    private Map<Integer, Object> lastDetection;
    private List<String> listOfResults = CameraActivity.listOfResults;
    private Context context;

    public JavaCameraDetectionView(Context context, int cameraId) {
        super(context, cameraId);
    }

    public JavaCameraDetectionView(Context context, AttributeSet attrs) {
        super(context, attrs);
        this.context = context;
    }

    protected void deliverAndDrawFrame(CvCameraViewFrame frame) {
        Mat modified = frame.rgba();

        if (mListener != null) {
            // get camera frame async
            if (detectionOutput == null) {
                detectionOutput = mListener.onCameraFrameAsync(frame);
            }
            else {
                // if recognition is done
                if (detectionOutput.isDone()){
                    try {
                        Object detection = detectionOutput.get();
                        if (detection instanceof Map) {
                            lastDetection = (Map<Integer, Object>) detectionOutput.get();
                        } else {
                            throw new IllegalArgumentException();
                        }
                    }
                    catch (ExecutionException | InterruptedException e) {
                        e.printStackTrace();
                    }
                    detectionOutput = null;
                }
            }

            // draw last prediction -> remove blinking effect
            if (lastDetection != null){
                // draw box on current frame, not frame used for detection
                try {
                    ObjectDetection objectDetection = new ObjectDetection(context.getAssets());
                    objectDetection.drawBoxes(lastDetection, modified, listOfResults, true);
                    // see results on main thread for UI
                    new Handler(Looper.getMainLooper()).post(() -> CameraActivity.adapter.notifyDataSetChanged());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        boolean bmpValid = true;
        if (modified != null) {
            try {
                Utils.matToBitmap(modified, mCacheBitmap);
            }
            catch(Exception e) {
                Log.e(TAG, "Mat type: " + modified);
                Log.e(TAG, "Bitmap type: " + mCacheBitmap.getWidth() + "*" + mCacheBitmap.getHeight());
                Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
                bmpValid = false;
            }
        }

        if (bmpValid && mCacheBitmap != null) {
            Canvas canvas = getHolder().lockCanvas();

            // camera preview portrait mode
            float mScale1;
            float mScale2;

            // get scale value
            if(canvas.getHeight()>canvas.getWidth()){
                canvas.rotate(90f,canvas.getWidth()/2,canvas.getHeight()/2);
                mScale1=(float)canvas.getHeight()/(float)mCacheBitmap.getWidth();
                mScale2=(float)canvas.getWidth()/(float)mCacheBitmap.getHeight();
            }
            else{
                mScale1=(float)canvas.getWidth()/(float)mCacheBitmap.getWidth();
                mScale2=(float)canvas.getHeight()/(float)mCacheBitmap.getHeight();
            }

            // scale frame for entire screen of phone
            if (canvas != null) {
                canvas.drawColor(0, android.graphics.PorterDuff.Mode.CLEAR);
                if (BuildConfig.DEBUG)
                    Log.d(TAG, "mStretch value: " + mScale);

                if (mScale1 != 0) {
                    canvas.drawBitmap(mCacheBitmap, new Rect(0,0,mCacheBitmap.getWidth(), mCacheBitmap.getHeight()),
                            new Rect((int)((canvas.getWidth() - mScale1*mCacheBitmap.getWidth()) / 2),
                                    (int)((canvas.getHeight() - mScale2*mCacheBitmap.getHeight()) / 2),
                                    (int)((canvas.getWidth() - mScale1*mCacheBitmap.getWidth()) / 2 + mScale1*mCacheBitmap.getWidth()),
                                    (int)((canvas.getHeight() - mScale2*mCacheBitmap.getHeight()) / 2 + mScale2*mCacheBitmap.getHeight())), null);
                }

                if (mFpsMeter != null) {
                    mFpsMeter.measure();
                    mFpsMeter.draw(canvas, 20, 30);
                }
                getHolder().unlockCanvasAndPost(canvas);
            }
        }
    }
}
