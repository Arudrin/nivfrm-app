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
import android.graphics.Point;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;

public class Regression {
    private static final String MODEL_FILE = "regression.tflite";
    private static final int BYTE_SIZE_OF_FLOAT = 4;

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

    private Regression() {
    }

    /**
     * Memory-map the model file in Assets.
     */
    private static ByteBuffer loadModelFile(AssetManager assets)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session.
     */
    public static Regression create(
            AssetManager assetManager,
            int inputWidth,
            int inputHeight,
            int sensorOrientation) {
        final Regression d = new Regression();

        try {
            GpuDelegate gpuDelegate = new GpuDelegate();
            Interpreter.Options options = new Interpreter.Options();
            options.addDelegate(gpuDelegate);
            d.tfLite = new Interpreter(loadModelFile(assetManager), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.sensorOrientation = sensorOrientation;
        d.width = inputWidth;
        d.height = inputHeight;

        // Pre-allocate buffers.
        d.intValues = new int[inputWidth * inputHeight];
        d.imgData = ByteBuffer.allocateDirect(inputWidth * inputHeight * BYTE_SIZE_OF_FLOAT);
        d.imgData.order(ByteOrder.nativeOrder());
        d.outputValues = new int[inputWidth * inputHeight];
        d.outputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 2);
        d.outputBuffer.order(ByteOrder.nativeOrder());

        d.pointStack = new Stack<>();
        d.maskStack = new Stack<>();
        return d;
    }

    Float estimate(List<Bitmap> bitmaps) {

        outputBuffer.rewind();
        Object[] inputs = new Object[30];

        for (int idx = 0; idx < 30; idx++) {
            imgData.rewind();
            Bitmap bitmap = bitmaps.get(idx);
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            for (final int val : intValues) {
                imgData.putFloat((val & 0xFF) / 255.0f);
            }
            inputs[idx] = imgData;
        }

        Log.d("TAG", "imgData: " + imgData.capacity());
        Log.d("TAG", "outputBuffer: " + outputBuffer.capacity());

        float[][] output = new float[1][1];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, output);

        // Copy the input data into TensorFlow.
        tfLite.runForMultipleInputsOutputs(inputs, outputs);

        float flowrate = ((float[][]) outputs.get(0))[0][0];
        Log.d("TAG", "flowrate: " + flowrate);

        return flowrate;
    }

}
