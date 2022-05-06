package com.example.trafficsignrecognitionandroidapp;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Toast;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class PickActivity extends AppCompatActivity {
    private String TAG = "Pick Activity";
    private String defaultText = "No image selected. Choose an image and the results will appear here.";
    private Button selectImage;
    private Button downloadResults;
    private ListView listView;
    private List<String> listOfResults = new ArrayList<>();
    private List<Integer> displayedSignClass = new ArrayList<>();
    private ArrayAdapter<String> adapter;
    private ImageView imageView;
    private SignRecognition signRecognition;
    private int SELECT_PICTURE = 200;

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
        downloadResults = findViewById(R.id.download_button);
        imageView = findViewById(R.id.image_view);

        // add placeholder for image container
        imageView.setImageDrawable(getResources().getDrawable(R.drawable.ic_baseline_image_search_24, null));

        // list of results frontend
        adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, listOfResults);
        listView = findViewById(R.id.pick_results_list);
        listView.setAdapter(adapter);

        // placeholder text
        listOfResults.add(0, defaultText);
        adapter.notifyDataSetChanged();

        // load model
        try {
            signRecognition = new SignRecognition(getAssets());
            Log.d(TAG, "Model is successfully loaded");
        } catch (IOException e) {
            Log.d(TAG, "Getting some error: " + e.getMessage());
        }

        // action for select image button
        selectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // choose image when button is clicked
                imageChooser();
            }
        });

        // action for download results button
        downloadResults.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // choose image when button is clicked
                downloadResultsOfRecognition();
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

        if(listOfResults.get(0).equals(defaultText)) {
            listOfResults.remove(0);
        }

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
                    image = signRecognition.detectionImage(image, listOfResults, displayedSignClass);

                    // convert image Mat to bitmap
                    Bitmap bitmapRecognize;
                    bitmapRecognize = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(image, bitmapRecognize);

                    // set image to image view
                    imageView.setImageBitmap(bitmapRecognize);

                    // notify for new results
                    String result = "Image name: \n" + getFileName(selectedImageUri) + "\n\n" + listOfResults.get(0);
                    listOfResults.remove(0);
                    listOfResults.add(0, result);
                    adapter.notifyDataSetChanged();
                }
            }
        }
    }

    private void downloadResultsOfRecognition() {
        if(listOfResults.get(0).equals(defaultText)) {
            Toast.makeText(getApplicationContext(), "To download you must make at least one recognition.", Toast.LENGTH_SHORT).show();
        }
        else {
            try {
                File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "Traffic sign recognition");
                if (!file.exists()) {
                    file.mkdir();
                }
                File gpxfile = new File(file, "traffic-sign-recognition-results-" + new SimpleDateFormat("dd-MM-yyyy-HH-mm-ss", Locale.getDefault()).format(new Date()) + ".txt");
                FileWriter writer = new FileWriter(gpxfile);
                for (String result : listOfResults) {
                    writer.append(result).append("\n-----------------------------------\n");
                }
                writer.flush();
                writer.close();
                Toast.makeText(getApplicationContext(), "File saved successfully in " + Environment.DIRECTORY_DOWNLOADS + "/Traffic sign recognition", Toast.LENGTH_SHORT).show();
            } catch (IOException e) {
                Toast.makeText(getApplicationContext(), "An error occurs. Try again", Toast.LENGTH_SHORT).show();
                e.printStackTrace();
            }
        }
    }

    private String getFileName(Uri uri) throws IllegalArgumentException {
        // Obtain a cursor with information regarding this uri
        Cursor cursor = getContentResolver().query(uri, null, null, null, null);

        // check cursor length
        if (cursor.getCount() <= 0) {
            cursor.close();
            throw new IllegalArgumentException("Can't obtain file name, cursor is empty");
        }

        cursor.moveToFirst();
        String fileName = cursor.getString(cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME));
        cursor.close();

        // return real filename
        return fileName;
    }

}