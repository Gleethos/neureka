package ut.tensors

import neureka.Tsr
import spock.lang.Specification

class Tensor_State_Unit_Test extends Specification
{

    def 'Newly instantiated and unmodified tensors have expected state.'()
    {
        given : Tsr t = new Tsr(6);
        expect : !t.isOutsourced();
        when : t.setIsOutsourced(true);
        then :
            t.isOutsourced();
            !t.is64() && !t.is32();
            t.value64()==null;
            t.value32()==null;
        when : t.setIsOutsourced(false);
        then :
            t.value64()!=null;
            t.isVirtual();
        when : t = new Tsr(new int[]{2}, 5);
        then : !t.isOutsourced();
        when : t.setIsOutsourced(true);
        then :
            t.isOutsourced();
            !t.is64() && !t.is32();
            t.value64()==null;
            t.value32()==null;
        when : t.setIsOutsourced(false);
        then :
            t.value64()!=null;
            t.isVirtual();
    }


}
