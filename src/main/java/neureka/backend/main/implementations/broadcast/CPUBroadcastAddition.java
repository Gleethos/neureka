package neureka.backend.main.implementations.broadcast;

import neureka.backend.api.annotations.Loadable;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.implementations.fun.api.CPUBiFun;
import neureka.backend.main.operations.operator.Addition;

@Loadable(operation = Addition.class, algorithm = Broadcast.class)
public class CPUBroadcastAddition extends CPUBroadcast
{
    public CPUBroadcastAddition() {}

    @Override
    protected CPUBiFun _getFun() {
        return new CPUBiFun() {
            @Override public double invoke(double a, double b) { return a + b; }
            @Override public float invoke(float a, float b) { return a + b; }
            @Override public boolean invoke(boolean a, boolean b) { return a && b; }
            @Override public char invoke(char a, char b) { return (char) (((int)a)+((int)b)); }
        };
    }

    @Override protected CPUBiFun _getDeriveAt0() { return _getFun(); }

    @Override protected CPUBiFun _getDeriveAt1() { return _getFun(); }
}
