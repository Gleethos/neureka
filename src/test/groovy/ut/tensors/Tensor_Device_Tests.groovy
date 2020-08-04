package ut.tensors

import neureka.Tsr
import neureka.acceleration.Device
import spock.lang.Specification

class Tensor_Device_Tests extends Specification
{

    def 'Tensors try to migrate themselves to a device that is being added to them as component.'()
    {
        given : 'A mock device and a simple tensor instance.'
            def device = Mock(Device)
            Tsr t = new Tsr(2)
            device.has(t) >>> [false, true, true]

        when : 'The mock device is being added to the tensor...'
            t.add(device)

        then : '...the tensor should try to add itself to the given device.'
            1 * device.add(t)

        and : 'It "views itself" as outsourced.'
            t.isOutsourced()

        and : 'It stores the device as a component.'
            t.has(Device.class)
    }

    def 'Tensors try to remove themselves from their device when "setIsOutsourced(false)" is being called.'()
    {
        given : 'A simple tensor instance with a mock device as component.'
            def device = Mock(Device)
            device.has(_) >>> [false, true, true]
            Tsr t = new Tsr(2).add(device)

        when : 'The "isOutsourced" property is being set to false...'
            t.isOutsourced = false

        then : '...the tensor should try to remove itself from the given device.'
            1 * device.get(t)

        and : 'The device should not be a tensor component anymore.'
            !t.has(Device.class)
    }

    def 'The device of a tensor can be accessed via the "device()" method.'()
    {
        given : 'A simple tensor having a device as component'
            def device = Mock(Device)
            device.has(_) >>> [false, true, true]
            Tsr t = new Tsr(1).add(device)

        when : 'The device is being accessed via the "device()" method...'
            Device found = t.device()

        then : 'This found device should be the one that was set originally.'
            found == device
    }

    def 'When creating slices of tensors then this should trigger a "parent - child" relation noticable to the device!'()
    {
        given : 'A 2D tensor having a device as component'
            def device = Mock(Device)
            Tsr t = new Tsr([3, 3],[1, 2, 3, 4, 5, 6, 7, 8, 9])
            device.has(t) >>> [false, true]

        when : 'A slice is being created from the given tensor...'
            Tsr s = t[1..2, 0..2]

        and : 'The "parent tensor" is being migrated to the device...'
            t.add(device)

        then : '...this tensor should try to add itself to the given device.'
            1 * device.add(t)

        and : 'The child should become outsourced.'
            s.isOutsourced()

        and: 'Internally the Tsr "asked" if it belongs to the Device twice. (before and after migration attempt)'
            2 * device.has(t)
    }


}
