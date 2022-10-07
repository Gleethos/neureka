package ut.backend.core

import neureka.backend.api.ini.BackendRegistry
import neureka.backend.api.DeviceAlgorithm
import neureka.backend.api.Operation
import neureka.backend.api.ini.ImplementationReceiver
import neureka.devices.Device
import neureka.backend.ocl.CLBackend
import spock.lang.Specification

class OpenCL_Backend_Spec extends Specification
{
    def 'The OpenCL backend context can load implementations.'() {
        given: 'A backend context is being created...'
            var ctx = new CLBackend()
        and: 'A backend loader is created by the context...'
            var loader = ctx.getLoader()
        expect :
            loader != null

        and : 'A mocked receiver for implementations is being created...'
            var receiver = Mock(ImplementationReceiver)
            var registry = BackendRegistry.of(receiver)
        when : 'The context is being asked to load implementations...'
            loader.load(registry)
        then:
            (12..666) * receiver.accept(
                    { Operation.isAssignableFrom(it) },
                    { DeviceAlgorithm.isAssignableFrom(it) },
                    { Device.isAssignableFrom(it) },
                    { it != null }
                    )
    }

}
