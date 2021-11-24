package neureka.backend.api;

import neureka.internal.common.composition.Component;
import neureka.devices.Device;

/**
 *  Implementations of this might introduce CUDA or ROCM to Neureka.
 *  By default, this interface is used to implement an OpenCL context
 *  via the {@link neureka.devices.opencl.CLContext} class used by the standard backend.
 *  If you want to introduce new backends to Neureka, this is the place to start!
 */
public interface BackendExtension extends Component<Extensions> {

    /**
     *  The {@link BackendContext} does not handle {@link Device} instances directly.
     *  Instead, the task of instantiating and exposing {@link Device} implementations
     *  should be carried by {@link BackendExtension} implementations.
     *  One extension might be implementing CUDA operations,
     *  therefore, the extension should also deal with some sort of CUDA{@link Device} implementation.
     *
     * @param searchKey The search key used to find a suitable {@link Device} implementation in this extension.
     * @return A suitable {@link DeviceOption} or null if nothing was found.
     */
    DeviceOption find( String searchKey );

    /**
     *  Tells this extension to dispose itself.
     *  One should not use a {@link BackendExtension} after it was disposed!
     */
    void dispose();


    class DeviceOption {

        private final Device<?> _device;
        private final double _confidence;

        public DeviceOption( Device<?> device, double confidence ) {
            _device = device;
            _confidence = confidence;
        }

        public Device<?> device() { return _device; }

        public double confidence() { return _confidence; }

    }

}
