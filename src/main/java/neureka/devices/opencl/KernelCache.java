package neureka.devices.opencl;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 *  A fixed sized cache for ad-hoc (just in time compiled) {@link OpenCLDevice} kernels.
 *  This cache will mostly only be utilized when integrating with custom kernels
 *  or when {@link neureka.backend.api.Operation}s are being optimized for
 *  the {@link OpenCLDevice}. <br> <br>
 *  <b>Warning: This is an internal class, meaning it should not be used
 *  anywhere but within this library. <br>
 *  This class or its public methods might change or get removed in future versions!</b>
 */
public final class KernelCache {

    private final static int CAPACITY = 256;

    private final Map<String, OpenCLDevice.cl_ad_hoc> _adhocKernels =
    new LinkedHashMap<String, OpenCLDevice.cl_ad_hoc>(CAPACITY) {
        @Override
        protected boolean removeEldestEntry(final Map.Entry eldest) {
            return size() > CAPACITY;
        }
    };

    public void put( String name, OpenCLDevice.cl_ad_hoc kernel ) {
        // Storing the ad hoc object in a fixed size map for fast access by operations:
        _adhocKernels.put( name, kernel );
    }

    public boolean has( String name ) {
        return _adhocKernels.containsKey( name );
    }

    public OpenCLDevice.cl_ad_hoc get( String name ) {
        return _adhocKernels.get( name );
    }

}
