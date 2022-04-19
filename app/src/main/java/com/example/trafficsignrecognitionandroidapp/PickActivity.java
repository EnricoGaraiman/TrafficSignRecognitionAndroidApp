package com.example.trafficsignrecognitionandroidapp;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

public class PickActivity extends AppCompatActivity {
    private String TAG = "Pick Activity";
    private Button selectImage;
    private ImageView imageView;
    private ObjectDetection objectDetection;
    int SELECT_PICTURE = 200;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pick);

        // initialize nav
        BottomNavigationView bottomNavigationView = findViewById(R.id.bottom_navigation);

        // set home selected
        bottomNavigationView.setSelectedItemId(R.id.pick);

        // perform ItemSelectedListener
        bottomNavigationView.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem item) {
                switch (item.getItemId()) {
                    case R.id.pick:
                        return true;
                    case R.id.realtime:
                        startActivity(new Intent(getApplicationContext(), CameraActivity.class));
                        overridePendingTransition(0, 0);
                        return true;
                    case R.id.home:
                        startActivity(new Intent(getApplicationContext(), MainActivity.class));
                        overridePendingTransition(0, 0);
                        return true;
                }

                return false;

            }
        });

        // define btn image
        selectImage = findViewById(R.id.select_button);
        imageView = findViewById(R.id.image_view);

        // load model
        try {
            objectDetection = new ObjectDetection(getAssets());
            Log.d(TAG, "Model is successfully loaded");
        } catch (IOException e) {
            Log.d(TAG, "Getting some error");
            e.printStackTrace();
        }

        // action for select image button
        selectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // choose image when button is clicked
                imageChooser();
            }
        });
    }

    private void imageChooser() {
        // create an instance of the
        // intent of the type image
        Intent i = new Intent();
        i.setType("image/*");
        i.setAction(Intent.ACTION_GET_CONTENT);

        // pass the constant to compare it
        // with the returned requestCode
        startActivityForResult(Intent.createChooser(i, "Select Picture"), SELECT_PICTURE);
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode==RESULT_OK) {
            if(requestCode==SELECT_PICTURE) {

                // image was selected with success
                Uri selectedImageUri = data.getData();
                if(selectedImageUri != null) {
                    Log.d(TAG, "onActivityResult: Image selected");

                    // read uri in Bitmap
                    Bitmap bitmap = null;
                    try {
                        bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // convert Bitmap to Mat
                    Mat image = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC4); // rgb
                    Utils.bitmapToMat(bitmap, image);

                    // send image to recognition method
                    image = objectDetection.detectionImage(image);

                    // convert image Mat to bitmap
                    Bitmap bitmapRecognize;
                    bitmapRecognize = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(image, bitmapRecognize);

                    // set image to image view
                    imageView.setImageBitmap(bitmapRecognize);
                }
            }
        }
    }

}