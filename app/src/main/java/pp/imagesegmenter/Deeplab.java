/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package pp.imagesegmenter;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Stack;

import pp.imagesegmenter.env.ImageUtils;

public class Deeplab {
    /**
     * An immutable result returned by a Deeplap describing what was recognized.
     */
    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        private Bitmap bitmap;

        Recognition(
                final String id, final RectF location, final Bitmap bitmap) {
            this.id = id;
            this.location = location;
            this.bitmap = bitmap;
        }

        public String getId() {
            return id;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public Bitmap getBitmap() {return bitmap;}

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    private static final String MODEL_FILE = "segmentation.tflite";
    // Float model
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    private static final int BYTE_SIZE_OF_FLOAT = 4;

    private static final int[] colormap = {
            0x00000000,     //background
            0x99ffe119,     //aeroplane
            0x993cb44b,     //bicycle
            0x99808000,     //bird
            0x99008080,     //boat
            0x99000080,     //bottle
            0x99e6194b,     //bus
            0x99f58230,     //car
            0x99800000,     //cat
            0x99d2f53c,     //chair
            0x99aa6e28,     //cow
            0x9946f0f0,     //diningtable
            0x99911eb4,     //dog
            0x99f032e6,     //horse
            0x990082c8,     //motobike
            0x99fabebe,     //person
            0x99ffd7b4,     //pottedplant
            0x99808080,     //sheep
            0x99fffac8,     //sofa
            0x99aaffc3,     //train
            0x99e6beff      //tv
    };

    private int sensorOrientation;
    private int width;
    private int height;

    private int[] intValues;
    private ByteBuffer imgData;
    private ByteBuffer outputBuffer;
    private int[] outputValues;

    private Stack<Point> pointStack;
    private Stack<Point> maskStack;

    private Interpreter tfLite;

    /** Memory-map the model file in Assets. */
    private static ByteBuffer loadModelFile(AssetManager assets)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** Initializes a native TensorFlow session. */
    public static Deeplab create(
            AssetManager assetManager,
            int inputWidth,
            int inputHeight,
            int sensorOrientation) {
        final Deeplab d = new Deeplab();

        try {
            GpuDelegate delegate = new GpuDelegate();
            Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
            d.tfLite = new Interpreter(loadModelFile(assetManager), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.sensorOrientation = sensorOrientation;
        d.width = inputWidth;
        d.height = inputHeight;

        // Pre-allocate buffers.
        d.intValues = new int[inputWidth * inputHeight];
        d.imgData = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * BYTE_SIZE_OF_FLOAT);
        d.imgData.order(ByteOrder.nativeOrder());
        d.outputValues = new int[inputWidth * inputHeight];
        d.outputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 2);
        d.outputBuffer.order(ByteOrder.nativeOrder());

        d.pointStack = new Stack<>();
        d.maskStack = new Stack<>();
        return d;
    }


    private Deeplab() {}

    Bitmap segment(Bitmap bitmap, Matrix matrix) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        outputBuffer.rewind();

        for (final int val : intValues) {
            imgData.putFloat((float) (((val >> 16) & 0xFF) - 103.939));
            imgData.putFloat((float) (((val >> 8) & 0xFF) - 116.779));
            imgData.putFloat((float) ((val & 0xFF) - 123.68));
        }

        Log.e("TAG", "imgData: " + imgData.capacity());
        Log.e("TAG", "outputBuffer: " + outputBuffer.capacity());

        // Copy the input data into TensorFlow.
        tfLite.run(imgData, outputBuffer);

        outputBuffer.flip();

        int white = Color.rgb(255, 255, 255);
        int black = Color.rgb(0, 0, 0);

        int customHeight = 120, customWidth = 120;
        Bitmap bgBitmap = Bitmap.createBitmap(customWidth, customHeight, Bitmap.Config.ARGB_8888);
        Bitmap streamBitmap = Bitmap.createBitmap(customWidth, customHeight, Bitmap.Config.ARGB_8888);

        for (int row = 0; row < customHeight; row++) {
            for (int col = 0; col < customWidth; col++) {
                for (int idx = 0; idx < 2; idx++) {
                    float value = outputBuffer.getFloat();
                    if (idx % 2 == 0) {
                        bgBitmap.setPixel(col, row, value > 0.99 ? white : black);
                    } else {
                        streamBitmap.setPixel(col, row, value > 0.99 ? white : black);
                    }
                }
            }
        }

//        ImageUtils.saveBitmap(bgBitmap, "bg.png");
//        ImageUtils.saveBitmap(streamBitmap, "stream.png");

//        Matrix maskMatrix = new Matrix();
//        ImageUtils.getTransformationMatrix(
//                120, 120,
//                240, 240,
//                sensorOrientation, false).invert(maskMatrix);
//        int[] rgbBytes = new int[640 * 480];
//        Bitmap rgbFrameBitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
//        rgbFrameBitmap.setPixels(rgbBytes, 0, 640, 0, 0, 640, 480);
//
//        Canvas canvas = new Canvas(streamBitmap);
//        canvas.drawBitmap(rgbFrameBitmap, maskMatrix, null);

        return streamBitmap;
    }

    private Bitmap createMask(int id, Matrix matrix, RectF rectF) {
        int w = (int) rectF.width();
        int h = (int) rectF.height();

        int top = (int) rectF.top;
        int left = (int) rectF.left;

        Bitmap segBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        int[] tmpValues = new int[w * h];

        while (!maskStack.empty()) {
            Point point = maskStack.pop();
            tmpValues[(point.y - top) * w + point.x - left] = colormap[id];
        }

        segBitmap.setPixels(tmpValues, 0, w, 0, 0, w, h);
        matrix.mapRect(rectF);

        Bitmap mask = Bitmap.createBitmap((int) rectF.width(), (int) rectF.height(), Bitmap.Config.ARGB_8888);

        Matrix maskMatrix = new Matrix();
        ImageUtils.getTransformationMatrix(
                (int) rectF.width(), (int) rectF.height(),
                w, h,
                sensorOrientation, false).invert(maskMatrix);

        Canvas canvas = new Canvas(mask);
        canvas.drawBitmap(segBitmap, maskMatrix, null);

        return mask;
    }

    private void floodFill(int initX, int initY, int val, RectF rectF) {
        outputValues[initY * width + initX] = 0;
        pointStack.push(new Point(initX, initY));

        while (!pointStack.empty()) {
            Point point = pointStack.pop();
            maskStack.push(point);

            int row = point.x;
            int col = point.y;

            if (rectF.top > col) rectF.top = col;
            if (rectF.bottom < col + 1) rectF.bottom = col + 1;
            if (rectF.left > row) rectF.left = row;
            if (rectF.right < row + 1) rectF.right = row + 1;

            if (row > 0 && val == outputValues[col * width + row - 1]) {
                outputValues[col * width + row - 1] = 0;
                pointStack.push(new Point(row - 1, col));
            }
            if (row < width - 1 && val == outputValues[col * width + row + 1]) {
                outputValues[col * width + row + 1] = 0;
                pointStack.push(new Point(row + 1, col));
            }
            if (col > 0 && val == outputValues[(col - 1) * width + row]) {
                outputValues[(col - 1) * width + row] = 0;
                pointStack.push(new Point(row, col - 1));
            }
            if (col < height - 1 && val == outputValues[(col + 1) * width + row]) {
                outputValues[(col + 1) * width + row] = 0;
                pointStack.push(new Point(row, col + 1));
            }
        }
    }
}
