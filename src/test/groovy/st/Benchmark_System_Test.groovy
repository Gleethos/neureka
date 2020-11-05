package st

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.frontend.Cache
import neureka.devices.Device
import neureka.devices.host.HostCPU
import org.slf4j.Logger
import spock.lang.Specification
import testutility.Utility

import java.text.SimpleDateFormat

class Benchmark_System_Test extends Specification
{

    def 'Tensor can be constructed by passing List instances.'()
    {
        given : 'Neureka instance is being reset.'
            Neureka.instance().reset()

        when : Tsr t = new Tsr([1, 3, 6])
        then :
            assert !t.toString().contains("empty")
            assert t.toString().contains("(1x3x6)")

        when : t = new Tsr([1, 3.0, 6])
        then :
            assert !t.toString().contains("empty")
            assert t.toString().contains("(1x3x6):[0.0, 0.0, 0.0")
        when : t = new Tsr([1, 3.3, 6])
        then :
            assert !t.toString().contains("empty")
            assert t.toString().contains("(3):[1.0, 3.3, 6.0]")
    }

    def 'Test benchmark script and simple tensor constructor.'()
    {
        given : 'Neureka instance is being reset.'
            Neureka.instance().reset()
            def configuration = [ "iterations":1, "sample_size":20, "difficulty":15, "intensifier":0 ]

        and : 'Function cache mocking is being prepared to test logging...'
            Logger oldLogger = Function.CACHE._logger
            Function.CACHE._logger = Mock( Logger )

        and : 'The benchmark script is being loaded into a GroovyShell instance.'
            def session = new GroovyShell().evaluate(Utility.readResource("benchmark.groovy", this))

        and : 'A String instance for the result hash is being instantiated and the expected hash.'
            String hash = ""
            String expected = "56b2eb74955e49cd777469c7dad0536e"

        when : 'The benchmark script is being called...'
            session(
                    configuration, null,
                    HostCPU.instance(),
                    tsr -> {
                        hash = (hash+tsr.toString()).md5()
                    }
            )

        then : 'The hash is as expected.'
            hash == expected

        and : 'No logging occurs because the benchmark does not render a scenario where a cache hit could occur.'
            0 * Function.CACHE._logger.debug(_)

        when : 'The cache logging is being reverted to the original state...'
            Function.CACHE._logger = oldLogger
        and : 'Only continue if testing system supports OpenCL.'
            if ( !Neureka.instance().canAccessOpenCL() ) return

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
                    HostCPU.instance(),
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
                    HostCPU.instance(),
                    tsr->{}
            )
            session([
                        "iterations":1,
                        "sample_size":20,
                        "difficulty":15,
                        "intensifier":50
                    ],
                    "neureka0.4.1_CPU_it1_ss20_dif_15_int50_"+currentDate+".csv",
                    HostCPU.instance(),
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
                                Tsr t = new Tsr([difficulty,difficulty], -5..9)
                                t.forEach( n -> n )
                            }
                    }]
                ],
                "neureka_1_CPU_it1_ss20_dif_15_int50_"+currentDate+".csv",
                HostCPU.instance(),
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
                    HostCPU.instance(),
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
                    HostCPU.instance(),
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
                    HostCPU.instance(),
                    tsr->{}
            )
         */
    }


}
