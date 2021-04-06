package neureka.utility;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class RangeInterpreter {

    private final List<Integer> _steps = new ArrayList<>();
    private final List<Object> _ranges = new ArrayList<>();

    public RangeInterpreter( Object[] ranges )
    {
        for (Object range : ranges ) {
            if ( range instanceof Map) {
                _ranges.addAll(((Map<?, ?>) range).keySet());
                _steps.addAll(((Map<?, Integer>) range).values());
            }
            else if ( range instanceof int[] ) {
                List<Integer> intList = new ArrayList<>(((int[]) range).length);
                for ( int ii : (int[]) range ) intList.add(ii);
                _ranges.add(intList);
                _steps.add(1);
            } else if ( range instanceof String[] ) {
                List<String> strList = new ArrayList<>(((String[]) range).length);
                strList.addAll(Arrays.asList((String[]) range));
                _ranges.add(strList);
                _steps.add(1);
            }
            else {
                _ranges.add( range );
                _steps.add(1);
            }
        }
    }

    public Object[] getRanges() {
        return _ranges.toArray();
    }

    public int[] getSteps() {
        return _steps.stream().mapToInt( s -> s ).toArray();
    }




}
