/*
* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package pp.imagesegmenter;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.util.Size;
import android.util.TypedValue;
import android.widget.FrameLayout;
import android.widget.ImageView;

import com.google.android.material.snackbar.Snackbar;

import java.util.ArrayList;
import java.util.Vector;

import pp.imagesegmenter.env.BorderedText;
import pp.imagesegmenter.env.ImageUtils;
import pp.imagesegmenter.env.Logger;
import pp.imagesegmenter.tracking.MultiBoxTracker;

import static java.lang.Thread.sleep;

/**
* An activity that uses a Deeplab and ObjectTracker to segment and then track objects.
*/
public class MainActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final int CROP_SIZE = 240;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    private Deeplab deeplab;
    private Regression regression;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    private Snackbar initSnackbar;
    private ImageView maskView;
    private ImageView extractedView;


    private boolean initialized = false;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        FrameLayout container = findViewById(R.id.container);
        initSnackbar = Snackbar.make(container, "Initializing...", Snackbar.LENGTH_INDEFINITE);

        init();

        final float textSizePx =
        TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        CROP_SIZE, CROP_SIZE,
                        sensorOrientation, true);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    tracker.draw(canvas);
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }
                });

        addCallback(
                canvas -> {
                    if (!isDebug()) {
                        return;
                    }
                    final Bitmap copy = cropCopyBitmap;
                    if (copy == null) {
                        return;
                    }

                    final int backgroundColor = Color.argb(100, 0, 0, 0);
                    canvas.drawColor(backgroundColor);

                    final Matrix matrix = new Matrix();
                    final float scaleFactor = 2;
                    matrix.postScale(scaleFactor, scaleFactor);
                    matrix.postTranslate(
                            canvas.getWidth() - copy.getWidth() * scaleFactor,
                            canvas.getHeight() - copy.getHeight() * scaleFactor);
                    canvas.drawBitmap(copy, matrix, new Paint());

                    final Vector<String> lines = new Vector<String>();
                    lines.add("Frame: " + previewWidth + "x" + previewHeight);
                    lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                    lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                    lines.add("Rotation: " + sensorOrientation);
                    lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                    borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                });

        maskView = findViewById(R.id.maskView);
        extractedView = findViewById(R.id.extractedView);
    }

    OverlayView trackingOverlay;

    void init() {
        runInBackground(() -> {
            runOnUiThread(()->initSnackbar.show());
            try {
                deeplab = Deeplab.create(getAssets(), CROP_SIZE, CROP_SIZE, sensorOrientation);
            } catch (Exception e) {
                LOGGER.e("Exception initializing classifier!", e);
                finish();
            }
            runInBackground(() -> {
                regression = Regression.create(getAssets(), CROP_SIZE, CROP_SIZE, sensorOrientation);
                runOnUiThread(() -> initSnackbar.dismiss());
                initialized = true;
            });

        });
    }

    final ArrayList<Bitmap> extractedStreams = new ArrayList<>(30);

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        tracker.onFrame(previewWidth, previewHeight, getLuminanceStride(), sensorOrientation, originalLuminance, timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection || !initialized) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        runInBackground(() -> {
            final Canvas canvas = new Canvas(croppedBitmap);
            canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

            final Bitmap streamMask = deeplab.segment(croppedBitmap);
            final Bitmap scaledMask = Bitmap.createScaledBitmap(streamMask, 240, 240, false);
            final Bitmap extractedStream = extractWaterStream(croppedBitmap, scaledMask);
            extractedStreams.add(extractedStream);

            final int streamsCollected = extractedStreams.size();
            runOnUiThread(() -> {
                initSnackbar.setText("Collected " + streamsCollected + " frames...");
                initSnackbar.show();
                maskView.setImageBitmap(scaledMask);
                extractedView.setImageBitmap(extractedStream);
            });

            if (streamsCollected < 30) {
                trackingOverlay.postInvalidate();
                requestRender();
                computingDetection = false;
                return;
            }

            final Float flowrate = regression.segment(extractedStreams);
            extractedStreams.clear();

            runOnUiThread(() -> {
                initSnackbar.setText(flowrate.toString());
                initSnackbar.show();
            });

            try {
                sleep(5000);
            } catch (InterruptedException e) {
                // can't sleep :(
                e.printStackTrace();
            }

            trackingOverlay.postInvalidate();
            requestRender();
            computingDetection = false;
        });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private Bitmap extractWaterStream(Bitmap streamImage, Bitmap streamMask) {
        final int black = Color.rgb(0, 0, 0);

        for (int row = 0; row < streamImage.getHeight(); row++) {
            for (int col = 0; col < streamImage.getWidth(); col++) {
                int maskPixel = streamMask.getPixel(col, row);
                if (maskPixel == black) streamImage.setPixel(col, row, black);
            }
        }

        return streamImage;
    }
}
