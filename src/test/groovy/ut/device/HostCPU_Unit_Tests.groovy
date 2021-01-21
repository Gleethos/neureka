package ut.device

import neureka.Neureka
import neureka.Tsr;
import neureka.devices.Device;
import neureka.devices.host.HostCPU
import neureka.utility.TsrAsString
import spock.lang.Specification;

class HostCPU_Unit_Tests extends Specification
{
    def setupSpec()
    {
        reportHeader """
                <h2> HostCPU Behavior </h2>
                <br> 
                <p>
                    The thread pool of the HostCPU executor becomes
                    more active when receiving larger workloads which
                    benefit from parallelization.           
                </p>
            """
    }

    def setup() {
        Neureka.instance().reset()
        // Configure printing of tensors to be more compact:
        Neureka.instance().settings().view().asString = TsrAsString.configFromCode("dgc")
    }


    def "thread pool executes given workload in parallel"()
    {
        given :
            Tsr a = new Tsr(new int[]{100, 60, 1, 2}, 4)
            Tsr b = new Tsr(new int[]{100, 1, 60, 2}, -2)
            Device cpu = a.getDevice()
            assert cpu!=null
            HostCPU.NativeExecutor exec = ((HostCPU)cpu).getExecutor()
            assert exec!=null
            assert exec.getPool() != null

        expect :
            if(exec.getPool().getCorePoolSize()<=2) true
            else {
                int[] min = new int[]{ exec.getPool().getCorePoolSize() }
                assert min[0] == Runtime.getRuntime().availableProcessors();
                Thread t = new Thread(() -> {
                    while (min[0] > 0) {
                        int current = exec.getPool().getCorePoolSize() - exec.getPool().getActiveCount();
                        if (current < min[0]) min[0] = current;
                    }
                })
                t.start()
                Tsr c = (a / b) * 3
                assert c.shape() == [100,60,60,2]
                int result = min[0]
                try {
                    min[0] = 0
                    t.join()
                } catch (InterruptedException e) {
                    e.printStackTrace()
                }
                assert result <= (exec.getPool().getCorePoolSize() / 2)
            }

    }


}
