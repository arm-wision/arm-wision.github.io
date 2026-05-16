\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ceurart}[2026/05/16 Local CEURART compatibility class]

\LoadClass[twocolumn]{article}

\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{xparse}
\RequirePackage{authblk}
\RequirePackage{geometry}
\RequirePackage{natbib}

\geometry{margin=0.75in}
\setlength{\columnsep}{0.25in}

\makeatletter
\newcommand\ceur@conference{}
\newcommand\ceur@copyrightclause{}
\newcommand\ceur@abstract{}
\newcommand\ceur@keywords{}
\newcommand\copyrightyear[1]{\gdef\ceur@copyrightyear{#1}}
\newcommand\copyrightclause[1]{\gdef\ceur@copyrightclause{#1}}
\newcommand\conference[1]{\gdef\ceur@conference{#1}}

\renewcommand\maketitle{%
  \twocolumn[%
    \begin{@twocolumnfalse}%
      \begin{center}%
        {\LARGE\bfseries \@title\par}%
        \vskip 1em%
        {\large \@author\par}%
        \vskip 1em%
        \ifx\ceur@conference\@empty\else {\small \ceur@conference\par}\vskip .5em\fi%
        \ifx\ceur@copyrightclause\@empty\else {\footnotesize \ceur@copyrightclause\par}\vskip 1em\fi%
      \end{center}%
      \ifx\ceur@abstract\@empty\else
        \begin{center}%
          \begin{minipage}{0.94\textwidth}%
            \noindent\textbf{Abstract. }\ceur@abstract\par%
          \end{minipage}%
        \end{center}%
        \vskip 0.75em%
      \fi%
      \ifx\ceur@keywords\@empty\else
        \begin{center}%
          \begin{minipage}{0.94\textwidth}%
            \noindent\textbf{Keywords: }\ceur@keywords\par%
          \end{minipage}%
        \end{center}%
        \vskip 1em%
      \fi%
    \end{@twocolumnfalse}%
  ]%
}
\makeatother

\let\ceurorigauthor\author
\RenewDocumentCommand{\author}{O{} m O{}}{\ceurorigauthor[#1]{#2}}
\NewDocumentCommand{\address}{O{} m}{\affil[#1]{#2}}
\NewDocumentCommand{\fnmark}{O{} }{}
\NewDocumentCommand{\fntext}{O{} m}{}

\makeatletter
\RenewDocumentEnvironment{abstract}{+b}{\gdef\ceur@abstract{#1}}{}
\NewDocumentEnvironment{keywords}{+b}{\gdef\ceur@keywords{#1}}{}
\makeatother
\newcommand\sep{; }
\NewDocumentEnvironment{acknowledgments}{+b}{%
  \section*{Acknowledgments}#1%
}{}

\bibliographystyle{plainnat}
