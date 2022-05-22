package com.example.trafficsignrecognitionandroidapp;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TableLayout;
import android.widget.TableRow;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "CameraActivity";

    private Mat mRgba;
    private CameraBridgeViewBase mOpenCvCameraView;
    private SignRecognition signRecognition;
    private ListView listView;
    private TableLayout table;
    private TableRow tableRow;
    private int displayedRecognizedSignPreview = 5;
    public static List<String> listOfResults = new ArrayList<>();
    public static List<Integer> displayedSignClass = new ArrayList<>();
    public static ArrayAdapter<String> adapterResults;
    public boolean lockPreview = false;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface
                        .SUCCESS: {
                    Log.i(TAG, "Opencv loaded: ");
                    mOpenCvCameraView.enableView();
                }
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public CameraActivity() {
        Log.i(TAG, "CameraActivity: Instantiate new" + this.getClass());
    }

    public void setLockPreview(boolean lock) {
        lockPreview = lock;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSION_REQUEST_CAMERA = 0;
        int MY_PERMISSION_REQUEST_STORAGE = 0;

        // request permission user camera
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(CameraActivity.this, new String[]{Manifest.permission.CAMERA}, MY_PERMISSION_REQUEST_CAMERA);
        }

        // request permission user storage
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(CameraActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, MY_PERMISSION_REQUEST_STORAGE);
        }

        setContentView(R.layout.activity_camera);

        // initialize nav
        BottomNavigationView bottomNavigationView = findViewById(R.id.bottom_navigation);

        // set home selected
        bottomNavigationView.setSelectedItemId(R.id.realtime);

        // perform ItemSelectedListener
        bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem item) {
                switch (item.getItemId()) {
                    case R.id.realtime:
                        return true;
                    case R.id.home:
                        startActivity(new Intent(getApplicationContext(), MainActivity.class));
                        overridePendingTransition(0, 0);
                        return true;
                    case R.id.pick:
                        startActivity(new Intent(getApplicationContext(), PickActivity.class));
                        overridePendingTransition(0, 0);
                        return true;
                }

                return false;

            }
        });

        mOpenCvCameraView = findViewById(R.id.frame_surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableFpsMeter(); // fps
        mOpenCvCameraView.setMaxFrameSize(640, 640); // max frame size improve FPS

        // get model
        try {
            signRecognition = new SignRecognition(getAssets());
            Log.d(TAG, "Model is successfully loaded");
        } catch (IOException e) {
            Log.d(TAG, "Getting some error");
            e.printStackTrace();
        }

        // list of results frontend
        adapterResults = new ArrayAdapter<>(this, R.layout.list_item, listOfResults);
        listView = findViewById(R.id.real_time_results);
        listView.setDivider(null);
        listView.setAdapter(adapterResults);

        // table for preview recognized signs
        table = findViewById(R.id.table_layout_preview_signs);
        tableRow = (TableRow) table.getChildAt(0);
    }

    @Override
    protected void onResume() {
        super.onResume();

        // check Opencv load status
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "Opencv: initialized success ");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "Opencv: initialized fail");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        // mRgba = inputFrame.rgba();
        //
        // // recognize
        // Mat out = new Mat();
        // ObjectDetection.drawBoxes(objectDetection.recognizeFrame(mRgba), mRgba);
        //
        // return out;
        return null; // i implement async detection with function onCameraFrameAsync
    }

    @Override
    public CompletableFuture<?> onCameraFrameAsync(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Log.d(TAG, "onCameraFrameAsync: Get frame and send to detection");

        // get frame for detection on separate thread -> unblocking UI
        return CompletableFuture.supplyAsync(() -> {

            // set detected signs on layout in main thread + results
            if(!lockPreview) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            setLockPreview(true);

                            // see results on main thread for UI
                            notifyRecognitionResultsChanged();

                            // preview signs
                            setRecognizedSignsPreviewOnLayout();

                            // lock preview
                            setLockPreview(false);
                        }
                        catch (Exception e) {
                            Log.e(TAG, "UI thread run: " + e.getMessage());
                        }
                    }
                });
            }

            // return next frame
            mRgba = inputFrame.rgba();
            return signRecognition.detectionFrame(mRgba);
        });
    }

    private void notifyRecognitionResultsChanged() {
        if(listOfResults.size() != 0) {
            adapterResults.notifyDataSetChanged();
        }
    }

    private void setRecognizedSignsPreviewOnLayout() {
        int imageResource;
        ImageView recognizedSign;
        Drawable res;

        // remove duplicate
        List<Integer> displayedSignClassSet = new ArrayList<>(new HashSet<>(displayedSignClass));

        // get first displayedRecognizedSignPreview recognized sign preview
        for (int c = 0; c < displayedRecognizedSignPreview; c++) {
            recognizedSign = (ImageView) tableRow.getChildAt(c);

            if (c < displayedSignClassSet.size() && displayedSignClassSet.iterator().hasNext()) {
                // get image resource based on recognized class
                imageResource = getResources().getIdentifier("@drawable/sign_class_" + displayedSignClassSet.get(c), null, getPackageName());
                res = getResources().getDrawable(imageResource, null);
                recognizedSign.setImageDrawable(res);
            }
            else {
                // remove preview image
                if(recognizedSign != null) {
                    recognizedSign.setImageDrawable(null);
                }
            }
        }

        // clear displayedSignClass
        displayedSignClass.clear();
    }
}