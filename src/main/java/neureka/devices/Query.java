package neureka.devices;

import neureka.Neureka;
import neureka.backend.api.BackendExtension;
import neureka.calculus.assembly.ParseUtil;
import neureka.devices.host.CPU;

import java.util.Arrays;
import java.util.stream.Stream;

class Query
{
    private static final double ACCEPTABILITY = 0.42;
    private static final String[] TAKE_FIRST = {"first device","first", "first gpu", "first cpu", "primary", "main", "any", "anything", "something"};
    private static final String[] WANTS_GPU = {"first gpu", "gpu", "nvidia", "amd", "intel", "opencl", "fpga", "radeon", "cuda", "apu", "graphics", "rdna", "rocm", "graphics"};
    private static final String[] WANTS_CPU = {"first cpu", "jvm","native","host","cpu","threaded", "processor", "main processor", "central processor", "central processing unit"};

    private Query() {}


    static <T, D extends Device<T>> D query( Class<D> deviceType, String... searchKeys ) {
        String[] flattened =
            Arrays.stream(searchKeys)
                .flatMap( key -> Arrays.stream(key.split(" or ")) )
                .flatMap( key -> Arrays.stream(key.split("\\|\\|")) )
                .map(String::trim)
                .filter( key -> !key.isEmpty() )
                .flatMap( key -> key.equals("amd") ? Stream.of("amd", "advanced micro devices") : Stream.of(key) )
                .flatMap( key -> key.equals("nvidia") ? Stream.of("nvidia", "nvidia corporation") : Stream.of(key) )
                .toArray(String[]::new);

        return queryInternal( deviceType, flattened );
    }


    private static <T, D extends Device<T>> D queryInternal( Class<D> deviceType, String... searchKeys )
    {
        if ( deviceType == CPU.class ) return (D) CPU.get();
        String key;
        if ( searchKeys.length == 0 ) key = "";
        else key = String.join(" ", searchKeys).toLowerCase();

        boolean justTakeFirstOne = Arrays.asList(TAKE_FIRST).contains(key);
        boolean probablyWantsGPU = Arrays.stream(WANTS_GPU).anyMatch(key::contains);
        probablyWantsGPU = probablyWantsGPU || key.equals("first") || key.equals("first device");

        double desireForCPU = Arrays.stream(WANTS_CPU)
                                        .flatMapToDouble(
                                            cpuWord ->
                                                Arrays.stream(searchKeys)
                                                        .mapToDouble(word -> ParseUtil.similarity( word, cpuWord ) )
                                        )
                                        .max()
                                        .orElse(0);

        if ( probablyWantsGPU ) desireForCPU /= 10; // CPU instance is most likely not meant!

        for ( String currentKey : searchKeys )
            for ( BackendExtension extension : Neureka.get().backend().getExtensions() ) {
                BackendExtension.DeviceOption found = extension.find( currentKey );
                if ( found == null           ) continue;
                if ( found.device() == null  ) continue;
                if ( found.confidence() <= 0 ) continue;
                if ( !deviceType.isAssignableFrom( found.device().getClass() ) ) continue;
                if ( found.confidence() > ACCEPTABILITY && found.confidence() > desireForCPU || (justTakeFirstOne && probablyWantsGPU) )
                    return (D) found.device();
            }

        if ( probablyWantsGPU )
            return null; // User wants OpenCL but cannot have it :/
        else if ( deviceType.isAssignableFrom( CPU.class ) && (desireForCPU > ACCEPTABILITY || justTakeFirstOne) )
            return (D) CPU.get();
        else
            return null; // We don't know what the user wants, but we do not have it :/
    }


}
