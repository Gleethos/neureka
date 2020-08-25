package st

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.acceleration.host.HostCPU
import spock.lang.Specification
import testutility.Utility

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

        and : 'The benchmark script is being loaded into a GroovyShell instance.'
            def session = new GroovyShell().evaluate(Utility.readResource("benchmark.groovy", this))

        and : 'A String instance for the result hash is being instantiated and the expected hash.'
            String hash = ""
            String expected = "56b2eb74955e49cd777469c7dad0536e"

        when : 'The benchmark script is being called...'
            session([
                    "iterations":1,
                    "sample_size":20,
                    "difficulty":15,
                    "intensifier":0
            ],
                    null,
                    HostCPU.instance(),
                    tsr -> {
                        hash = (hash+tsr.toString()).md5()
                    }
            )

        then : 'The hash is as expected.'
            hash == expected

        and : 'If system supports OpenCL.'
            if ( !Neureka.instance().canAccessOpenCL() ) return

        when : 'The benchmark is now being executed with the first found OpenCLDevice instance...'
            hash = ""
            session([
                    "iterations":1,
                    "sample_size":20,
                    "difficulty":15,
                    "intensifier":0
            ],
                    null,
                    Device.find("first"),
                        tsr -> {
                            hash = (hash+tsr.toString()).md5()
                        }
            )

        then : 'The calculated hash is as expected.'
            hash==expected

        //String currentDate = new SimpleDateFormat("dd-MM-yyyy").format(new Date())
        //session([
        //            "iterations":1,
        //            "sample_size":20,
        //            "difficulty":15,
        //            "intensifier":50
        //        ],
        //        "neureka_bench_GPU_"+currentDate+".csv",
        //        Device.find("nvidia"),
        //        tsr->{}
        //)
        //session([
        //            "iterations":1,
        //            "sample_size":20,
        //            "difficulty":15,
        //            "intensifier":50
        //        ],
        //        "neureka_bench_CPU_"+currentDate+".csv",
        //        HostCPU.instance(),
        //        tsr->{}
        //)
        //session([
        //            "iterations":1,
        //            "sample_size":100,
        //            "difficulty":500,
        //            "intensifier":0
        //        ],
        //        "neureka_bench_GPU_100x_cd100_"+currentDate+".csv",
        //        Device.find("nvidia"),
        //        tsr->{}
        //)
        //session([
        //            "iterations":1,
        //            "sample_size":500,
        //            "difficulty":5,
        //            "intensifier":0
        //        ],
        //        "neureka_bench_CPU_500x_cd5_"+currentDate+".csv",
        //        HostCPU.instance(),
        //        tsr->{}
        //)

        // NDIM - BENCHMARK :
        //Neureka.instance().settings().ndim().setIsOnlyUsingDefaultNDConfiguration(true)
        //session([
        //            "iterations":1,
        //            "sample_size":250,
        //            "difficulty":10,
        //            "intensifier":0
        //        ],
        //        "ndim_default_bench_CPU_250x_cd10_"+currentDate+".csv",
        //        HostCPU.instance(),
        //        tsr->{}
        //)
        //Neureka.instance().settings().ndim().setIsOnlyUsingDefaultNDConfiguration(false)
        //session([
        //        "iterations":1,
        //        "sample_size":250,
        //        "difficulty":10,
        //        "intensifier":0
        //],
        //        "ndim_optimized_bench_CPU_250x_cd10_"+currentDate+".csv",
        //        HostCPU.instance(),
        //        tsr->{}
        //)
    }


}
