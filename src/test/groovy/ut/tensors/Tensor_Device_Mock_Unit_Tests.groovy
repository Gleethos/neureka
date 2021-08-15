package ut.tensors

import neureka.Component
import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import spock.lang.Specification

class Tensor_Device_Mock_Unit_Tests extends Specification
{
    def setupSpec() {
        reportHeader """    
            <h2>Tensor Device Mock Tests</h2>
            <p>
                This unit test specification covers 
                the expected behavior of tensors when interacting
                with instances of implementations of the Device interface.
            </p>
        """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'Tensors try to migrate themselves to a device that is being added to them as component.'()
    {
        given : 'A mock device and a simple tensor instance.'
            def device = Mock(Device)
            Tsr t = Tsr.of(2)
            device.has(t) >>> [false, true, true]

        when : 'The mock device is being added to the tensor...'
            t.set(device)

        then : '...the tensor should not try to add itself to the given device via the "store" method.'
            0 * device.store(t)

        and : 'Instead it should use the "update" method, which is a formal callback from the internal component system...'
            1 * device.update({ it.type().name() == "ADDED" }) >> true

        and : 'It stores the device as a component.'
            t.has(Device.class)
    }

    def 'Tensors try to remove themselves from their device when "setIsOutsourced(false)" is being called.'()
    {
        given : 'A simple tensor instance with a mock device as component.'
            def device = Mock(Device)
            device.has(_) >>> [false, true, true, false]
            device.update(_) >> true
            Tsr t = Tsr.of(2).set(device)

        when : 'The "isOutsourced" property is being set to false...'
            t.isOutsourced = false

        then : '...the tensor should try to remove itself from the given device.'
            (1.._) * device.restore( t )

        and : 'The device should not be a tensor component anymore.'
            !t.has(Device.class)
    }

    def 'The device of a tensor can be accessed via the "device()" method.'()
    {
        given : 'A simple tensor having a device as component'
            def device = Mock(Device)
            device.has(_) >>> [false, true, true] // Some realistic return values to simulate tensor reception!
            Tsr t = Tsr.of(1)

        when :
            t.set(device)
            t.setIsOutsourced(true)

        then :
            1 * device.update({
                Component.OwnerChangeRequest request -> request.executeChange()
            })

        when : 'The device is being accessed via the "device()" method...'
            Device found = t.get(Device.class)

        then : 'This found device should be the one that was set originally.'
            found == device

        and :
            found == t.getDevice()

        and :
            t.isOutsourced()
    }

    def 'When creating slices of tensors then this should trigger a "parent - child" relation noticeable to the device!'()
    {
        given : 'A 2D tensor having a device as component'
            def device = Mock(Device)
            Tsr t = Tsr.of([3, 3],[1, 2, 3, 4, 5, 6, 7, 8, 9])
            device.has(t) >>> [false, true]

        when : 'A slice is being created from the given tensor...'
            Tsr s = t[1..2, 0..2]

        and : 'The "parent tensor" is being migrated to the device...'
            t.set(device)

        then : '...this tensor should not try to add itself to the given device via the "store" method.'
            0 * device.store(t)

        and : 'Instead the "update" method should be called...'
            1 * device.update(_)

        and: 'Internally the Tsr may "asks" if it belongs to the Device. (before and after migration attempt)'
            (0.._) * device.has(t)
    }


}
