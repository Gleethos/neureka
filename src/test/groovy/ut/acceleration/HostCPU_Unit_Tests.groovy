package ut.acceleration

import neureka.Neureka
import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.HostCPU
import spock.lang.Specification;

class HostCPU_Unit_Tests extends Specification
{
    /**
     * The thread pool of the HostCPU executor becomes
     * more active when receiving larger workloads which
     * benefit from parallelization.
     */
    def "thread pool executes given workload in parallel"()
    {
        given :
            Neureka.instance().reset()
            Tsr a = new Tsr(new int[]{100, 60, 2}, 4)
            Tsr b = new Tsr(new int[]{100, 2, 60}, -2)
            Device cpu = a.device()
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
