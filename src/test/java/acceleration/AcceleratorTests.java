package acceleration;

import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.opencl.OpenCLPlatform;
import org.junit.Test;
import testutility.UnitTester_Tensor;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class AcceleratorTests
{

    @Test
    public void test_gpu_IO(){

        Neureka.instance().reset();
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