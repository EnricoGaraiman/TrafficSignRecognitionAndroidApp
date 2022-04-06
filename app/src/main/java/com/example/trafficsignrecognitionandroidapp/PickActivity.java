package com.example.trafficsignrecognitionandroidapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

public class PickActivity extends AppCompatActivity {
    private String TAG = "Pick Activity";
    private Button select_image;
    private ImageView image_view;
    private ObjectDetection objectDetection;
    int SELECT_PICTURE = 200;

    private String pathModel = "test2.tflite";
    private String pathLabels = "labelmap.txt";
    private int modelInputSize = 320;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pick);

        // define btn image
        select_image = findViewById(R.id.select_button);
        image_view = findViewById(R.id.image_view);

        // load model
        try {
            objectDetection = new ObjectDetection(getAssets(), pathModel, pathLabels, modelInputSize);
            Log.d(TAG, "Model is successfully loaded");
        } catch (IOException e) {
            Log.d(TAG, "Getting some error");
            e.printStackTrace();
        }

        // action for select image button
        select_image.setOnClickListener(new View.OnClickListener() {
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
                    image = objectDetection.recognizePhoto(image);

                    // convert image Mat to bitmap
                    Bitmap bitmap_recognize = null;
                    bitmap_recognize = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(image, bitmap_recognize);

                    // set image to image view
                    image_view.setImageBitmap(bitmap_recognize);
                }
            }
        }
    }

}