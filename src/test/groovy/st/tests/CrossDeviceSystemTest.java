package st.tests;

import neureka.Shape;
import neureka.Tensor;
import neureka.devices.Device;
import neureka.devices.opencl.OpenCLDevice;
import testutility.Sleep;
import testutility.UnitTester_Tensor;

import java.util.ArrayList;
import java.util.List;

public class CrossDeviceSystemTest
{

    public static boolean on( Device gpu )
    {
        int initialNumberOfOutsourced = gpu.numberOfStored();

        UnitTester_Tensor tester = new UnitTester_Tensor("");

        List<Tensor> listOfTensors = new ArrayList<>();
        {
            Tensor<Double> tensor1, tensor2;
            //=====================================================================
            tensor1 = Tensor.of(Shape.of(2, 2), new double[]{
                    -1, 7,
                    -2, 3,
            }).setRqsGradient(true);
            tensor2 = Tensor.of(Shape.of(2, 2), new double[]{
                    -1, 7,
                    -2, 3,
            });
            gpu.store(tensor1).store(tensor2);
            listOfTensors.add(tensor1);
            listOfTensors.add(tensor2);

            tester.testTensorAutoGrad(
                    new Tensor[]{tensor1, tensor2},
                    "I[0]*i1",
                    new String[]{"[2x2]:(1.0, 49.0, 4.0, 9.0)"});
            tester.testTensorAutoGrad(
                    new Tensor[]{tensor1, tensor2},
                    "I[0]xi1",
                    new String[]{
                            "[1x1]:(63.0)",
                            "[2x2]:(-1.0, 7.0, -2.0, 3.0) ]|:t{ [2x2]:(-1.0, 7.0, -2.0, 3.0) }"
                    });
            //=====================================================================
            tensor1 = Tensor.of(Shape.of(3, 5), new double[]{
                    2, 3, 5,
                    -4, 6, 2,
                    -5, -2, -1,
                    2, 4, -1,
                    1, 2, 7
            });
            gpu.store(tensor1);
            listOfTensors.add(tensor1);
            tester.testTensorAutoGrad(
                    new Tensor[]{tensor1}, "lig(I[0])",
                    new String[]{
                            "[3x5]:(2.12693, 3.04859, 5.00672, 0.01814, 6.00248, 2.12693, 0.00671, 0.12692, 0.31326, 2.12693, 4.01815, 0.31326, 1.31326, 2.12693, 7.00091)"
                    });
            //===================
            tensor1 = Tensor.of(Shape.of(2), 3d);
            tensor2 = Tensor.of(Shape.of(2), 4d);
            gpu.store(tensor1).store(tensor2);
            listOfTensors.add(tensor1);
            listOfTensors.add(tensor2);
            //result = Tensor.of(new Tensor[]{tensor1, tensor2}, "i0*i1");
            tester.testTensorAutoGrad(
                    new Tensor[]{tensor1, tensor2}, "i0*i1",
                    new String[]{"[2]:(12.0, 12.0)"});
            //===========================================
            tensor1 = Tensor.of(Shape.of(2, 3, 1),
                    new double[]{
                            3, 2,
                            -1, -2,
                            2, 4
                    }
            );
            tensor2 = Tensor.of(Shape.of(1, 3, 2),
                    new double[]{
                            4, -1, 3,
                            2, 3, -1
                    });
            gpu.store(tensor1);
            gpu.store(tensor2);
            listOfTensors.add(tensor1);
            listOfTensors.add(tensor2);
            tester.testTensorAutoGrad(
                    new Tensor[]{tensor1, tensor2}, "I0 x i1",
                    new String[]{"[2x1x2]:(15.0, 2.0, 10.0, 2.0)"});
            //=======================
            tensor1 = Tensor.of(Shape.of(200, 300, 1),  2d);
            tensor2 = Tensor.of(Shape.of(1, 300, 200), 3d);
            gpu.store(tensor1);
            gpu.store(tensor2);
            listOfTensors.add(tensor1);
            listOfTensors.add(tensor2);
            tester.testTensorAutoGrad(
                    new Tensor[]{tensor1, tensor2}, "I0xi1",
                    new String[]{"[200x1x200]:(1800.0, 1800.0, 1800.0, 1800.0, 1800.0, 1800.0,"});
            //---
            tensor1 = Tensor.of(Shape.of(2, 2, 1), new double[]{
                    1, 2, //  3, 1,
                    2, -3, // -2, -1,
            }).setRqsGradient(true);
            tensor2 = Tensor.of(Shape.of(1, 2, 2), new double[]{
                    -2, 3, //  0  7
                    1, 2,  // -7  0
            });
            gpu.store(tensor1).store(tensor2);
            listOfTensors.add(tensor1);
            listOfTensors.add(tensor2);
            tester.testTensorAutoGrad(//4, 5, -13, -4 <= result values
                    new Tensor[]{tensor1, tensor2},
                    "i0xi1",
                    new String[]{
                            "[2x1x2]:(" +
                                    "0.0, 7.0, -7.0, 0.0" +
                                    "); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"
                    },
                    Tensor.of(Shape.of(2, 1, 2), new double[]{1, 1, 1, 1}),
                    new double[][]{new double[]{1.0, 3.0, 1.0, 3.0}, null}
            );

            // ---
            Tensor<Double> x = Tensor.of(Shape.of(1), 3d).setRqsGradient(true);
            Tensor<Double> b = Tensor.of(Shape.of(1), -4d);
            Tensor<Double> w = Tensor.of(Shape.of(1), 2d);
            gpu.store(x).store(b).store(w);
            listOfTensors.add(x);
            listOfTensors.add(b);
            listOfTensors.add(w);
            /*
             *      ((3-4)*2)**2 = 4
             *  dx:   8*3 - 32  = -8
             */
            Tensor<Double> y = Tensor.of("((i0+i1)*i2)**2", x, b, w);
            tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0)"});
            y.backward(Tensor.of(2d));
            tester.testTensor(x, new String[]{"-16.0"});
            tester.testShareDevice(gpu, new Tensor[]{y, x, b, w});
            //---
            x = Tensor.of(Shape.of(1), 4d).setRqsGradient(true);
            b = Tensor.of(Shape.of(1), 0.5);
            w = Tensor.of(Shape.of(1), 0.5);
            gpu.store(x).store(b).store(w);
            listOfTensors.add(x);
            listOfTensors.add(b);
            listOfTensors.add(w);
            y = Tensor.of("(2**i0**i1**i2**2", x, b, w);
            tester.testTensor(y, new String[]{"[1]:(9.24238);", " ->d[1]:(28.4928)", " ->d[1]:(4.3207"});
            tester.testShareDevice(gpu, new Tensor[]{y, x, b, w});

            //====
            x = Tensor.of(Shape.of(1), 3d);
            b = Tensor.of(Shape.of(1), -5d);
            w = Tensor.of(Shape.of(1), -2d);
            gpu.store(x).store(b).store(w);
            listOfTensors.add(x);
            listOfTensors.add(b);
            listOfTensors.add(w);
            Tensor z = Tensor.of("I0*i1*i2", x, b, w);
            tester.testTensor(z, new String[]{"[1]:(30.0)"});
            tester.testShareDevice(gpu, new Tensor[]{z, x, b, w});

            //---
            x = Tensor.of(Shape.of(1), 3d).setRqsGradient(true);
            b = Tensor.of(Shape.of(1), 0.5);
            w = Tensor.of(Shape.of(1), 4d);
            gpu.store(x).store(b).store(w);
            listOfTensors.add(x);
            listOfTensors.add(b);
            listOfTensors.add(w);
            y = Tensor.of("(12/i0/i1/i2/2", x, b, w);//12/3/0.5/4/2 .... 12 * 1/ (0.0625)
            //listOfTensors.add(y);
            tester.testTensor(y, new String[]{"[1]:(1.0);", " ->d[1]:(-0.33333)"});
            tester.testShareDevice(gpu, new Tensor[]{y, x, b, w});
            //---
            //---------------------------------------------
            y = null;
            z = null;
            x = null;
            b = null;
            w = null;
            tensor1 = null;
            tensor2 = null;
        }
        listOfTensors.forEach((t)->t.setRqsGradient(false));//Removes gradients!

        if ( gpu instanceof OpenCLDevice )
        {
            System.gc();
            Sleep.until(400, ()->{
                int numberOfOutsourced = gpu.numberOfStored() - initialNumberOfOutsourced;
                return numberOfOutsourced <= listOfTensors.size();
            });
            System.gc();
            Sleep.until(400, ()->{
                int numberOfOutsourced = gpu.numberOfStored() - initialNumberOfOutsourced;
                return numberOfOutsourced <= listOfTensors.size();
            });

            int numberOfOutsourced = gpu.numberOfStored() - initialNumberOfOutsourced;

            int fix = gpu instanceof OpenCLDevice ? 1 : 0;

            assert numberOfOutsourced - fix <= listOfTensors.size();
            //---
            boolean[] stillOnDevice = new boolean[]{true};
            listOfTensors.forEach((t)->stillOnDevice[0] = gpu.contains(t) && stillOnDevice[0]);
            String sentence = "Used tensors still on device: ";
            tester.testContains(
                    sentence +stillOnDevice[0],
                    new String[]{sentence+"true"},
                    "Testing for memory leaks!"
            );
            //---
            listOfTensors.forEach(gpu::free);
            sentence = "Number of tensors after deleting: ";
            tester.testContains(
                    sentence + Math.max(0, gpu.numberOfStored()-initialNumberOfOutsourced - fix),
                    new String[]{sentence+"0"},
                    "Testing if all tensors have been deleted!"
            );
            //---
        }
        return true;
    }


}
