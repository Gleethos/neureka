package st

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.common.utility.SettingsLoader
import neureka.view.NDPrintSettings
import spock.lang.Shared
import spock.lang.Specification
import testutility.Load

class Benchmark_System_Test extends Specification
{
    @Shared def oldStream

    def setup() {
        // The following is similar to Neureka.get().reset() however it uses a groovy script for library settings:
        SettingsLoader.tryGroovyScriptsOn(Neureka.get(), script -> new GroovyShell(getClass().getClassLoader()).evaluate(script))
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
        oldStream = System.err
        System.err = Mock(PrintStream)
    }

    def cleanup() {
        System.err = oldStream
    }


    def 'Tensor can be constructed by passing List instances.'()
    {
        when : var t = Tsr.ofDoubles().withShape(1, 3, 6 ).all(0)
        then :
            assert !t.toString().contains("empty")
            assert t.toString().contains("(1x3x6)")

        when : t = Tsr.ofDoubles().withShape(1, 3, 6).all(0)
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

        and : 'The benchmark script is being loaded into a GroovyShell instance.'
            def session = new GroovyShell().evaluate(Load.resourceAt("benchmark.groovy", this))

        and : 'A String instance for the result hash is being instantiated and the expected hash.'
            String hash = ""
            String expected = "7084fb60f29fcce1ac96c2bee9c37b37"

        when : 'The benchmark script is being called...'
            Map<String,List<Double>> result = session(
                                    configuration, null,
                                    CPU.get(),
                                    tsr -> {
                                        hash = (hash+tsr.toString()).md5()
                                    }
                            )

        then : 'The hash is as expected.'
            hash == expected

        and :
            result.keySet().toList() == ["convolutional_matrix_multiplication", "matrix_multiplication", "vector_multiplication", "manual_convolution", "tensor_math", "iterations", "difficulty"]
        and :
            result.values().every { it.size() == 21 && it.every { it > 0 } }

        when : 'Only continue if testing system supports OpenCL.'
            if ( !Neureka.get().canAccessOpenCLDevice() ) return

        and : 'The benchmark is now being executed with the first found OpenCLDevice instance...'
            hash = ""
            session(
                    configuration, null,
                    Device.get("first"),
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
