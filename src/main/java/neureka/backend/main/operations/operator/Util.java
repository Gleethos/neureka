package neureka.backend.main.operations.operator;

import neureka.Shape;

class Util {

    static boolean canBeBroadcast(Shape a, Shape b) {
        if ( a.size() != b.size() ) return false;
        boolean areEqual = a.equals(b);
        if ( areEqual ) return true;
        for ( int i = 0; i < a.size(); i++ )
            if ( a.get(i) != b.get(i) && a.get(i) != 1 && b.get(i) != 1 )
                return false;

        return true;
    }

}
