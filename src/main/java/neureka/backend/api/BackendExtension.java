package neureka.backend.api;

import neureka.backend.api.ini.BackendLoader;
import neureka.common.composition.Component;
import neureka.devices.Device;
import neureka.backend.ocl.CLBackend;

/**
 *  Implementations of this might introduce CUDA or ROCM to Neureka.
 *  By default, this interface is used to implement an OpenCL context
 *  via the {@link CLBackend} class used by the standard backend.
 *  If you want to introduce new backends to Neureka, this is the place to start!
 */
public interface BackendExtension extends Component<Extensions>
{
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

    BackendLoader getLoader();

    /**
     *  This class describes an available {@link Device} implementation found for a given {@link BackendExtension}.
     *  It exists because a typical {@link BackendExtension} will most likely also have a
     *  custom {@link Device} implementation exposing a specific API for executing tensors on them...
     */
    class DeviceOption
    {
        private final Device<?> _device;
        private final double _confidence;

        public DeviceOption( Device<?> device, double confidence ) {
            _device = device;
            _confidence = confidence;
        }

        /**
         * @return The device which fits a given key word best.
         */
        public Device<?> device() { return _device; }

        /**
         * @return The confidence level determining how well a given search key matches the wrapped device.
         */
        public double confidence() { return _confidence; }

    }

}
