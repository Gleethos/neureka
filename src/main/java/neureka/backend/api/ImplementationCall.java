package neureka.backend.api;

import neureka.Tsr;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public interface ImplementationCall {

    public Tsr<?>[] getTensors();

    <V, T extends Arg<V>> V getValOf( Class<T> argumentClass );

    public static class Builder
    {
        private Tsr<?>[] _tensors;
        private final Args _arguments = Args.of(
                                                    Arg.DerivIdx.of(-1),
                                                    Arg.VarIdx.of(-1)
                                            );

        private Builder(Tsr<?>[] tensors) { _tensors = tensors; }

        public ImplementationCall andArgs( List<Arg> arguments ) {
            for ( Arg argument : arguments ) _arguments.set(argument);
            return new ImplementationCall() {
                @Override public Tsr<?>[] getTensors() { return _tensors; }
                @Override
                public <V, T extends Arg<V>> V getValOf( Class<T> argumentClass ) {
                    return _arguments.valOf(argumentClass);
                }
            };
        }

        public ImplementationCall andArgs( Arg<?>... arguments ) {
            return andArgs(Arrays.stream(arguments).collect(Collectors.toList()));
        }
    }

}
