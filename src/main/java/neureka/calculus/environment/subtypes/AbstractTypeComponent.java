package neureka.calculus.environment.subtypes;


public class AbstractTypeComponent<CreatorType> implements TypeComponent
{
    protected String _operation;
    protected String _deriviation;
    protected CreatorType _creator;

    public AbstractTypeComponent(String operation, String deriviation, CreatorType creator)
    {
        _operation = operation;
        _deriviation = deriviation;
        _creator = creator;
    }

    public String getAsString(){
        return _operation;
    }
        public String getDeriviationAsString(){
        return _deriviation;
    }
        public CreatorType getCreator(){
        return _creator;
    }

}


