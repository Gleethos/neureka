package neureka.devices.opencl;

import java.util.HashMap;
import java.util.Map;

public class KernelCache {

    private final static int CAPACITY = 256;

    private final Map<String, OpenCLDevice.cl_ad_hoc> _adhocKernels = new HashMap<>(CAPACITY);
    private final String[] _adhocKernelRingBuffer = new String[ CAPACITY ];

    private int _ringIndex = 0;


    public void put( String name, OpenCLDevice.cl_ad_hoc kernel ) {
        // Storing the ad hoc object in a weak hash map for fast access by operations :
        _adhocKernels.put( name, kernel );
        // Storing the ad hoc object in a ring buffer to avoid immediate garbage collection :
        _ringIndex = ( _ringIndex + 1 ) % _adhocKernelRingBuffer.length;
        String old = _adhocKernelRingBuffer[ _ringIndex ];
        _adhocKernels.remove( old );
        _adhocKernelRingBuffer[ _ringIndex ] = name;
    }

    public boolean has( String name ) {
        return _adhocKernels.containsKey( name );
    }

    public OpenCLDevice.cl_ad_hoc get( String name ) {
        return _adhocKernels.get( name );
    }

}
