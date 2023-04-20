package ut.device

import neureka.Data
import neureka.Neureka
import neureka.Shape
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title
import testutility.Sleep

@Title("The CPU device, an API for CPU based execution")
@Narrative('''

    The CPU class, one of many implementations of the Device interface, 
    is simply supposed to be an API for dispatching threaded workloads onto the CPU.
    Contrary to other types of device, the CPU will host tensor data by default, simply
    because the tensors will be stored in RAM if no device was specified.

''')
@Subject([CPU, Device])
class CPU_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """ 
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
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
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
            Tsr a = Tsr.of(Shape.of(100, 60, 1, 2),  4)
            Tsr b = Tsr.of(Shape.of(100, 1, 60, 2), -2)

        and : 'The default device returned by the first tensor:'
            Device cpu = a.getDevice()
        expect : 'This device should not be null but be an instance of the CPU representative device type.'
            cpu != null
            cpu instanceof CPU

        when : 'Accessing the executor of the cpu device...'
            CPU.JVMExecutor exec = ( (CPU) cpu ).getExecutor()
        then : 'The executor is not null as well as its internal thread pool!'
            exec != null

        expect :
            if ( exec.getCorePoolSize() <= 2 ) true
            else {
                int[] min = new int[]{ exec.getCorePoolSize() }
                assert min[0] > 0 && min[0] <= Runtime.getRuntime().availableProcessors()
                Thread t = new Thread(() -> {
                    while ( min[0] > 0 ) {
                        int current = exec.getCorePoolSize() - exec.getActiveThreadCount()
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
                assert result <= (exec.getCorePoolSize() / 2)
            }
    }

    def 'CPU knows the current number of available processor cores!'()
    {
        expect :
            CPU.get().coreCount == Runtime.getRuntime().availableProcessors()
    }

    def 'The CPU exposes a non null API for executing workloads in parallel.'()
    {
        expect :
            CPU.get().executor != null
        and :
            CPU.get().executor.activeThreadCount >= 0
            CPU.get().executor.corePoolSize >= 0
            CPU.get().executor.completedTaskCount >= 0
    }

    def 'The CPU device will keep track of the amount of tensors it stores.'()
    {
        given : 'A CPU device instance.'
            CPU cpu = CPU.get()
        and : 'We note the initial amount of tensors stored on the CPU.'
            int initial = cpu.size()
            int initialDataObjects = cpu.numberOfDataObjects()

        when : 'We first create a data object...'
            var data = Data.of( 42, 73, 11, 7 )
        then : 'The CPU should not have stored any tensors yet.'
            cpu.size() == initial

        when : 'We create a tensor from the data object...'
            var t = Tsr.of( Shape.of(2, 2), data )
        then : 'The CPU should know about the existence of a new tensor.'
            CPU.get().size() == initial + 1
        and : 'The number of data objects stored on the CPU should also be increased.'
            CPU.get().numberOfDataObjects() == initialDataObjects + 1

        when : 'We create a new tensor from the first one...'
            var t2 = t * 2
        then : 'The CPU should know about the existence of a new tensor as well as the data objects.'
            CPU.get().size() == initial + 2
            CPU.get().numberOfDataObjects() == initialDataObjects + 2

        when : 'We however create a new reshaped version of the first tensor...'
            var t3 = t.reshape( 4 )
        then : 'The CPU should also know about the existence of a new tensor, but not a new data object.'
            CPU.get().size() == initial + 3
            CPU.get().numberOfDataObjects() == initialDataObjects + 2

        when : 'We delete the references to the tensors, and then give the GC some time to do its job...'
            t = null
            t2 = null
            t3 = null
            System.gc()
            Thread.sleep( 128 )
            Sleep.until(1028, {CPU.get().size() == initial})
        then : 'The CPU should have forgotten about the tensors.'
            CPU.get().size() == initial
        and : 'The CPU should have forgotten about the data objects as well.'
            CPU.get().numberOfDataObjects() == initialDataObjects
    }
}
