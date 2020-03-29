package neureka.framing;

import neureka.Component;
import neureka.Tsr;

public class ValueMap  implements Component<Tsr> {

    private final int[] _map;

    public ValueMap(Tsr owner){
        _map = new int[owner.size()];
        for (int i=0; i<_map.length; i++) _map[i] = owner.i_of_i(i);
    }

    @Override
    public void update(Tsr oldOwner, Tsr newOwner) {
        oldOwner.remove(this.getClass());
        newOwner.add(this);
    }

    public int[] getMap(){
        return _map;
    }
}
