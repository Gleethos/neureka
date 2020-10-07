package neureka.acceleration.storage;

import neureka.Tsr;
import neureka.dtype.NumericType;

import java.io.IOException;
import java.util.Iterator;

public interface FileHead
{

    <T> void persist(Iterator<T> data) throws IOException;

    Tsr<?> load() throws IOException;

}
