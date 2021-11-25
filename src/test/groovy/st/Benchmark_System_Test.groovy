package st

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.common.utility.SettingsLoader
import org.slf4j.Logger
import spock.lang.Specification
import testutility.Utility

class Benchmark_System_Test extends Specification
{

    def setup() {
        // The following is similar to Neureka.get().reset() however it uses a groovy script for library settings:
        SettingsLoader.tryGroovyScriptsOn(Neureka.get(), script -> new GroovyShell(getClass().getClassLoader()).evaluate(script))
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors = "dgc"
    }

    def 'Tensor can be constructed by passing List instances.'()
    {
        when : Tsr t = Tsr.ofShape(1, 3, 6 )
        then :
            assert !t.toString().contains("empty")
            assert t.toString().contains("(1x3x6)")

        when : t = Tsr.ofShape([1, 3.0, 6])
        then :
            assert !t.toString().contains("empty")
            assert t.toString().contains("(1x3x6):[0.0, 0.0, 0.0")
        when : t = Tsr.of([1, 3.3, 6])
        then :
            assert !t.toString().contains("empty")
            assert t.toString().contains("(3):[1.0, 3.3, 6.0]")
    }

    def 'Test benchmark script and simple tensor constructor.'()
    {
        given :
            def configuration = [ "iterations":1, "sample_size":20, "difficulty":15, "intensifier":0 ]

        and : 'Function cache mocking is being prepared to test logging...'
            Logger oldLogger = Neureka.get().backend().getFunctionCache()._log
            Neureka.get().backend().getFunctionCache()._log = Mock( Logger )

        and : 'The benchmark script is being loaded into a GroovyShell instance.'
            def session = new GroovyShell().evaluate(Utility.readResource("benchmark.groovy", this))

        and : 'A String instance for the result hash is being instantiated and the expected hash.'
            String hash = ""
            String expected = "56b2eb74955e49cd777469c7dad0536e"

        when : 'The benchmark script is being called...'
            session(
                    configuration, null,
                    CPU.get(),
                    tsr -> {
                        hash = (hash+tsr.toString()).md5()
                    }
            )

        then : 'The hash is as expected.'
            hash == expected

        and : 'No logging occurs because the benchmark does not render a scenario where a cache hit could occur.'
            0 * Neureka.get().backend().getFunctionCache()._log.debug(_)

        when : 'The cache logging is being reverted to the original state...'
            Neureka.get().backend().getFunctionCache()._log = oldLogger
        and : 'Only continue if testing system supports OpenCL.'
            if ( !Neureka.get().canAccessOpenCL() ) return

        and : 'The benchmark is now being executed with the first found OpenCLDevice instance...'
            hash = ""
            session(
                    configuration, null,
                    Device.find("first"),
                        tsr -> {
                            hash = ( hash + tsr.toString() ).md5()
                        }
            )

        then : 'The calculated hash is as expected.'
            hash == expected



        //String currentDate = new SimpleDateFormat("dd-MM-yyyy").format(new Date())
        /*
            session([
                        "iterations":1,
                        "sample_size":20,
                        "difficulty":15,
                        "intensifier":50
                    ],
                    "neureka_bench_GPU_"+currentDate+".csv",
                    Device.find("nvidia"),
                    tsr->{}
            )
            session([
                        "iterations":1,
                        "sample_size":20,
                        "difficulty":15,
                        "intensifier":50
                    ],
                    "neureka_bench_CPU_"+currentDate+".csv",
                    CPU.get(),
                    tsr->{}
            )
    */

    /* // Testing NDIterator vs array based iterator...
            when :
            session([
                        "iterations":1,
                        "sample_size":20,
                        "difficulty":15,
                        "intensifier":5
                    ],
                    null,//"neureka_bench_CPU_"+currentDate+".csv",
                    CPU.get(),
                    tsr->{}
            )
            session([
                        "iterations":1,
                        "sample_size":20,
                        "difficulty":15,
                        "intensifier":50
                    ],
                    "neureka0.4.1_CPU_it1_ss20_dif_15_int50_"+currentDate+".csv",
                    CPU.get(),
                    tsr->{}
            )
            then : true
        */


        /*
        // Testing ND-iteration
        session([
                    "iterations":1,
                    "sample_size":20,
                    "difficulty":15,
                    "intensifier":50,
                    "custom_code":[
                        "iterating":{
                        iterations, difficulty ->
                            iterations.times {
                                Tsr t = Tsr.of([difficulty,difficulty], -5..9)
                                t.forEach( n -> n )
                            }
                    }]
                ],
                "neureka_1_CPU_it1_ss20_dif_15_int50_"+currentDate+".csv",
                CPU.get(),
                tsr->{}
        ) == null
        */

        /*
            session([
                        "iterations":1,
                        "sample_size":100,
                        "difficulty":500,
                        "intensifier":0
                    ],
                    "neureka_bench_GPU_100x_cd100_"+currentDate+".csv",
                    Device.find("nvidia"),
                    tsr->{}
            )
            session([
                        "iterations":1,
                        "sample_size":500,
                        "difficulty":5,
                        "intensifier":0
                    ],
                    "neureka_bench_CPU_500x_cd5_"+currentDate+".csv",
                    CPU.get(),
                    tsr->{}
            )
        */

        // NDIM - BENCHMARK :
        /*
            Neureka.instance().settings().ndim().setIsOnlyUsingDefaultNDConfiguration(true)
            session([
                        "iterations":1,
                        "sample_size":250,
                        "difficulty":10,
                        "intensifier":0
                    ],
                    "ndim_default_bench_CPU_250x_cd10_"+currentDate+".csv",
                    CPU.get(),
                    tsr->{}
            )
            Neureka.instance().settings().ndim().setIsOnlyUsingDefaultNDConfiguration(false)
            session([
                    "iterations":1,
                    "sample_size":250,
                    "difficulty":10,
                    "intensifier":0
            ],
                    "ndim_optimized_bench_CPU_250x_cd10_"+currentDate+".csv",
                    CPU.get(),
                    tsr->{}
            )
         */
    }


}
