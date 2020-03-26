
import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.opencl.OpenCLPlatform;
import org.junit.Test;
import util.NTester_Tensor;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class AcceleratorTests
{
    @Test
    public void test_openCL_device_if_present()
    {
        Neureka.instance().settings().reset();
        Neureka.instance().settings().view().setLegacy(true);
        if(!System.getProperty("os.name").toLowerCase().contains("windows")){
            return;
        }
        NTester_Tensor tester = new NTester_Tensor("Testing autograd on GPU");

        Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0);
        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(true);

        Neureka.instance().settings().indexing().setLegacy(true);
        OpenCLPlatform.PLATFORMS().get(0).recompile();
        _test_autograd(gpu, tester, true);

        Neureka.instance().settings().indexing().setLegacy(false);
        OpenCLPlatform.PLATFORMS().get(0).recompile();
        _test_autograd(gpu, tester, false);

        Neureka.instance().settings().debug().setKeepDerivativeTargetPayloads(false);
        tester.close();
    }
    private void _test_autograd(Device gpu, NTester_Tensor tester, boolean legacyIndexing)
    {
        //gpu.add(new Tsr(new int[]{1000000}, 3));
        List<Tsr> listOfTensors = new ArrayList<>();
        Tsr tensor1, tensor2;
        //=====================================================================
        tensor1 = new Tsr(new int[]{2, 2}, new double[]{
                -1, 7,
                -2, 3,
        }).setRqsGradient(true);//ERROR!! ???
        tensor2 = new Tsr(new int[]{2, 2}, new double[]{
                -1, 7,
                -2, 3,
        });
        gpu.add(tensor1).add(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);

        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "I[0]*i1",
                new String[]{
                        "[2x2]:(1.0, 49.0, 4.0, 9.0)"
                });
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "I[0]xi1",
                new String[]{
                        "[1x1]:(63.0)",
                        "[2x2]:(-1.0, 7.0, -2.0, 3.0) ]|:t{ [2x2]:(-1.0, 7.0, -2.0, 3.0) }"
                });
        //=====================================================================
        tensor1 = new Tsr(new int[]{3, 5}, new double[]{
                 2,  3,  5,
                -4,  6,  2,
                -5, -2, -1,
                 2,  4, -1,
                 1,  2,  7
        });
        gpu.add(tensor1);
        listOfTensors.add(tensor1);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1},
                "lig(I[0])",
                new String[]{
                        "[3x5]:(2.12693E0, 3.04859E0, 5.00672E0, 0.01814E0, 6.00248E0, 2.12693E0, 0.00671E0, 0.12692E0, 0.31326E0, 2.12693E0, 4.01815E0, 0.31326E0, 1.31326E0, 2.12693E0, 7.00091E0)"
                });
        //===================
        tensor1 = new Tsr(new int[]{2}, 3);
        tensor2 = new Tsr(new int[]{2}, 4);
        gpu.add(tensor1).add(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);
        //result = new Tsr(new Tsr[]{tensor1, tensor2}, "i0*i1");
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "i0*i1",
                new String[]{"[2]:(12.0, 12.0)"});
        //===========================================
        tensor1 = new Tsr(
                new int[]{2, 3, 1},
                new double[]{
                        3, 2,
                        -1, -2,
                        2, 4
                }
        );
        tensor2 = new Tsr(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1, 3,
                        2, 3, -1
                });
        gpu.add(tensor1);
        gpu.add(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "I0 x i1",
                new String[]{
                        (legacyIndexing)?"[2x1x2]:(19.0, 22.0, 1.0, -6.0)":"(15.0, 2.0, 10.0, 2.0)"
                });
        //=======================
        tensor1 = new Tsr(
                new int[]{200, 300, 1},
                2
        );
        tensor2 = new Tsr(
                new int[]{1, 300, 200},
                3);
        gpu.add(tensor1);
        gpu.add(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);
        tester.testTensorAutoGrad(
                new Tsr[]{tensor1, tensor2},
                "I0xi1",
                new String[]{
                        "[200x1x200]:(1800.0, 1800.0, 1800.0, 1800.0, 1800.0, 1800.0,"//...
                });
        //---
        tensor1 = new Tsr(new int[]{2, 2, 1}, new double[]{
                1, 2, //3, 1,
                2, -3, //-2, -1,
        }).setRqsGradient(true);
        tensor2 = new Tsr(new int[]{1, 2, 2}, new double[]{
                -2, 3, // 0  7
                1, 2,  // -7  0
        });
        gpu.add(tensor1).add(tensor2);
        listOfTensors.add(tensor1);
        listOfTensors.add(tensor2);
        tester.testTensorAutoGrad(//4, 5, -13, -4 <= result values
                new Tsr[]{tensor1, tensor2},
                "i0xi1",
                new String[]{
                        "[2x1x2]:("
                                +((legacyIndexing)?"4.0, -13.0, 5.0, -4.0":"0.0, 7.0, -7.0, 0.0")+
                        "); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"
                },
                new Tsr(new int[]{2, 1, 2}, new double[]{1, 1, 1, 1}),
                (legacyIndexing)
                        ?new double[][]{{-1.0, -1.0, 5.0, 5.0}, null}
                        :new double[][]{{1.0, 3.0, 1.0, 3.0}, null}
        );
        //result = new Tsr(new Tsr[]{tensor1, tensor1}, "ig0<-i0");
        //tester.testContains(tensor1.toString("g"), new String[]{"test"}, "");

        // ---
        Tsr x = new Tsr(new int[]{1}, 3).setRqsGradient(true);
        Tsr b = new Tsr(new int[]{1}, -4);
        Tsr w = new Tsr(new int[]{1}, 2);
        gpu.add(x).add(b).add(w);
        listOfTensors.add(x);
        listOfTensors.add(b);
        listOfTensors.add(w);
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        Tsr y = new Tsr(new Tsr[]{x, b, w}, "((i0+i1)*i2)^2");
        tester.testTensor(y, new String[]{"[1]:(4.0); ->d[1]:(-8.0), "});
        y.backward(new Tsr(2));
        tester.testTensor(x, new String[]{"-16.0"});
        tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});
        //---
        x = new Tsr(new int[]{1}, 4).setRqsGradient(true);
        b = new Tsr(new int[]{1}, 0.5);
        w = new Tsr(new int[]{1}, 0.5);
        gpu.add(x).add(b).add(w);
        listOfTensors.add(x);
        listOfTensors.add(b);
        listOfTensors.add(w);
        y = new Tsr(new Tsr[]{x, b, w}, "(2^i0^i1^i2^2");
        tester.testTensor(y, new String[]{"[1]:(4.0);", " ->d[1]:(1.38629E0), "});
        tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});

        //====
        x = new Tsr(new int[]{1}, 3);
        b = new Tsr(new int[]{1}, -5);
        w = new Tsr(new int[]{1}, -2);
        gpu.add(x).add(b).add(w);
        listOfTensors.add(x);
        listOfTensors.add(b);
        listOfTensors.add(w);
        Tsr z = new Tsr(new Tsr[]{x, b, w}, "I0*i1*i2");
        tester.testTensor(z, new String[]{"[1]:(30.0)"});
        tester.testShareDevice(gpu, new Tsr[]{z, x, b, w});

        //---
        x = new Tsr(new int[]{1}, 3).setRqsGradient(true);
        b = new Tsr(new int[]{1}, 0.5);
        w = new Tsr(new int[]{1}, 4);
        gpu.add(x).add(b).add(w);
        listOfTensors.add(x);
        listOfTensors.add(b);
        listOfTensors.add(w);
        y = new Tsr(new Tsr[]{x, b, w}, "(12/i0/i1/i2/2");//12/3/0.5/4/2 .... 12 * 1/ (0.0625)
        //listOfTensors.add(y);
        tester.testTensor(y, new String[]{"[1]:(1.0);", " ->d[1]:(-0.33333E0), "});
        tester.testShareDevice(gpu, new Tsr[]{y, x, b, w});
        //---
        //---------------------------------------------
        y = null;
        z = null;
        x = null;
        b = null;
        w = null;
        tensor1 = null;
        tensor2 = null;
        listOfTensors.forEach((t)->t.setRqsGradient(false));//Removes gradients!
        System.gc();
        try { Thread.sleep(400); } catch (InterruptedException e) { e.printStackTrace(); }
        System.gc();
        try { Thread.sleep(400); } catch (InterruptedException e) { e.printStackTrace(); }
        Collection<Tsr> outsourced = gpu.tensors();

        String sentence = "Number of outsourced tensors: ";
        tester.testContains(
                sentence +(outsourced.size()),
                new String[]{sentence+(listOfTensors.size())},
                "Testing for memory leaks!"
        );
        //---
        boolean[] stillOnDevice = {true};
        listOfTensors.forEach((t)->stillOnDevice[0] = outsourced.contains(t)&&stillOnDevice[0]);
        sentence = "Used tensors still on device: ";
        tester.testContains(
                sentence +stillOnDevice[0],
                new String[]{sentence+"true"},//true
                "Testing for memory leaks!"
        );
        //---
        listOfTensors.forEach((t)->gpu.rmv(t));
        sentence = "Number of tensors after deleting: ";
        tester.testContains(
                sentence +gpu.tensors().size(),
                new String[]{sentence+"0"},
                "Testing if all tensors have been deleted!"
        );
        //---
    }

    @Test
    public void test_gpu_IO(){

        Neureka.instance().settings().reset();
        if(!System.getProperty("os.name").toLowerCase().contains("windows")){
            return;
        }
        Device gpu = Device.find("nvidia");

        Tsr t = new Tsr(new int[]{3, 2}, new double[]{2, 4, -5, 8, 3, -2});
        t.add(gpu);
        assert t.toString().contains("(3x2):[2.0, 4.0, -5.0, 8.0, 3.0, -2.0]");
        t.setValue32(new float[]{1, 1, 1, 1, 1});
        assert t.toString().contains("(3x2):[1.0, 1.0, 1.0, 1.0, 1.0, -2.0]");
        t.setValue32(new float[]{3, 5, 6});
        t.setValue32(new float[]{4, 2, 3});
        assert t.toString().contains("(3x2):[4.0, 2.0, 3.0, 1.0, 1.0, -2.0]");
        t.setValue64(new double[]{9, 4, 7, -12});
        t.setValue64(new double[]{-5, -2, 1});
        assert t.toString().contains("(3x2):[-5.0, -2.0, 1.0, -12.0, 1.0, -2.0]");
        t.setValue32(new float[]{22, 24, 35, 80});
        t.setValue64(new double[]{-1, -1, -1});
        assert t.toString().contains("(3x2):[-1.0, -1.0, -1.0, 80.0, 1.0, -2.0]");
    }

    @Test
    public void test_threaded_CPU_execution() {

        Tsr a = new Tsr(new int[]{100, 60, 1}, 4);
        Tsr b = new Tsr(new int[]{100, 1, 60}, -2);
        Device cpu = a.device();
        assert cpu!=null;
        HostCPU.NativeExecutor exec = ((HostCPU)cpu).getExecutor();
        assert exec!=null;
        assert exec.getPool()!=null;
        if(exec.getPool().getCorePoolSize()>2){
            int[] min = {exec.getPool().getCorePoolSize()};
            assert min[0]==Runtime.getRuntime().availableProcessors();
            Thread t = new Thread(()->{
                while (min[0]>0){
                    int current = exec.getPool().getCorePoolSize()-exec.getPool().getActiveCount();
                    if(current<min[0]) min[0] = current;
                }
            });
            t.start();
            Tsr c = a.div(b);
            int result = min[0];
            try {
                min[0] = 0;
                t.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            assert result<=(exec.getPool().getCorePoolSize()/2);
        }
    }





}