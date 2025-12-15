const OrdinaryFeature = @NamedTuple{
    data_pt_index::IntType, 
    feature_index::IntType, 
    contribution::FloatType
    }

# contribution at the nucleotide level
const BioSequenceContributionMonomer = @NamedTuple{
    data_pt_index::IntType,
    alphabet_index::IntType,
    position::IntType,
    contribution::FloatType, # TODO make it flexible later
}

# contribution at the code level (1st hidden layer)
const BioSequenceContributionCode = @NamedTuple{
    data_pt_index::IntType,
    filter_index::IntType,
    position::IntType,
    contribution::FloatType,
    mag::FloatType
}
