using BanzhafInference
using Documenter

DocMeta.setdocmeta!(BanzhafInference, :DocTestSetup, :(using BanzhafInference); recursive=true)

makedocs(;
    modules=[BanzhafInference],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="BanzhafInference.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/BanzhafInference.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/BanzhafInference.jl",
    devbranch="main",
)
