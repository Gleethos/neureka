package neureka.ndim.iterators;

import neureka.ndim.config.NDIterator;

import java.util.StringJoiner;
import java.util.stream.IntStream;

public abstract class AbstractNDIterator implements NDIterator
{

    @Override
    public String toString(){
        StringBuilder b = new StringBuilder();

        StringJoiner sj = new StringJoiner(",");
        StringJoiner finalSj1 = sj;
        IntStream.of( this.shape() ).forEach(x -> finalSj1.add(String.valueOf(x)));

        b.append("S["+sj.toString()+"];");
        sj = new StringJoiner(",");
        StringJoiner finalSj = sj;
        IntStream.of( this.get() ).forEach(x -> finalSj.add(String.valueOf(x)));
        b.append("I["+sj.toString()+"];");
        return b.toString();
    }


}
