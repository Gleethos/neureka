package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.view.TsrStringSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("The CPU device, an API for CPU based execution")
@Narrative('''

    The CPU class, one of many implementations of the Device interface, 
    is simply supposed to be an API for dispatching threaded workloads onto the CPU.
    Contrary to other types of device, the CPU will host tensor data by default, simply
    because the tensors will be stored in RAM if no device was specified.

''')
class CPU_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
                <h2> $CPU Behavior </h2>
                <br> 
                <p>
                    The thread pool of the $CPU executor becomes
                    more active when receiving larger workloads which
                    benefit from parallelization.           
                </p>
            """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }


    def 'Thread pool executes given workload in parallel'()
    {
        reportInfo """
            Warning! This test is flaky simply because it relies on the behaviour of threads
            which may or may not behave as expected. 
        """

        given : 'Two 4 dimensional tensor instances.'
            Tsr a = Tsr.of(new int[]{100, 60, 1, 2},  4)
            Tsr b = Tsr.of(new int[]{100, 1, 60, 2}, -2)

        and : 'The default device returned by the first tensor:'
            Device cpu = a.getDevice()
        expect : 'This device should not be null but be an instance of the CPU representative device type.'
            cpu != null
            cpu instanceof CPU
        when : 'Accessing the executor of the cpu device...'
            CPU.JVMExecutor exec = ( (CPU) cpu ).getExecutor()
        then : 'The executor is not null as well as its internal thread pool!'
            exec != null
            exec.getPool() != null

        expect :
            if ( exec.getPool().getCorePoolSize() <= 2 ) true
            else {
                int[] min = new int[]{ exec.getPool().getCorePoolSize() }
                assert min[0] == Runtime.getRuntime().availableProcessors()
                Thread t = new Thread(() -> {
                    while ( min[0] > 0 ) {
                        int current = exec.getPool().getCorePoolSize() - exec.getPool().getActiveCount()
                        if ( current < min[0] ) min[0] = current
                    }
                })
                t.start()
                Tsr c = ( a / b ) * 3
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


    def 'CPU knows the current number of available processor cores!'() {

        expect :
            CPU.get().coreCount == Runtime.getRuntime().availableProcessors()

    }


}
