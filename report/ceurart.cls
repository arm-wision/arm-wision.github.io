\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ceurart}[2026/05/19 Local CEURART compatibility class (single-column)]

\LoadClass{article}

\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{xparse}
\RequirePackage{authblk}
\RequirePackage{geometry}
\RequirePackage{natbib}
\RequirePackage{graphicx}
\RequirePackage{keyval}
\RequirePackage{hyperref}

\geometry{margin=1in}
\hypersetup{colorlinks=true, urlcolor=blue!55!black, linkcolor=red!50!black, citecolor=blue!55!black}

\makeatletter
% Storage for title-block / footer items
\newcommand\ceur@conference{}
\newcommand\ceur@copyrightclause{}
\newcommand\ceur@copyrightyear{}
\newcommand\ceur@abstract{}
\newcommand\ceur@keywords{}
\newcommand\ceur@fntext{}
\newcommand\ceur@emails{}
\newcommand\ceur@currentauthor{}

\newif\ifceur@firstemail
\ceur@firstemailtrue

\newcommand\copyrightyear[1]{\gdef\ceur@copyrightyear{#1}}
\newcommand\copyrightclause[1]{\gdef\ceur@copyrightclause{#1}}
\newcommand\conference[1]{\gdef\ceur@conference{#1}}

% Author options parser. Currently extracts only the email; orcid/url are
% silently accepted (the real CEUR class typesets them too but we just
% drop them in this stub).
\define@key{ceurauthor}{email}{%
  \ifceur@firstemail
    \ceur@firstemailfalse
    \protected@xdef\ceur@emails{%
      \noexpand\href{mailto:#1}{\nolinkurl{#1}} (\ceur@currentauthor)%
    }%
  \else
    \protected@xdef\ceur@emails{%
      \ceur@emails;\noexpand\ %
      \noexpand\href{mailto:#1}{\nolinkurl{#1}} (\ceur@currentauthor)%
    }%
  \fi
}
\define@key{ceurauthor}{orcid}{}
\define@key{ceurauthor}{url}{}

% Intercept \author and \address.
\let\ceurorigauthor\author
\RenewDocumentCommand{\author}{O{} m O{}}{%
  \ceurorigauthor[#1]{#2}%
  \gdef\ceur@currentauthor{#2}%
  \setkeys{ceurauthor}{#3}%
}
\NewDocumentCommand{\address}{O{} m}{\affil[#1]{#2}}

% \fnmark[N] is currently a no-op (the linkage between authors and their
% contribution note is implicit). \fntext[N]{text} captures the contribution
% note that gets emitted with a leading dagger in the page-1 footer.
\NewDocumentCommand{\fnmark}{O{1}}{}
\NewDocumentCommand{\fntext}{O{1} m}{\gdef\ceur@fntext{#2}}

\renewcommand\maketitle{%
  \begin{center}%
    {\LARGE\bfseries \@title\par}%
    \vskip 1em%
    {\large \@author\par}%
    \vskip 1em%
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
  % Page-1 footer: emit each item as an un-numbered footnote so they appear
  % at the bottom of page 1, separated from the body by the standard rule.
  % Order: conference, contribution-note (with dagger), emails, copyright.
  \begingroup
    \def\thefootnote{}%
    \def\@makefnmark{}%
    \ifx\ceur@conference\@empty\else
      \footnotetext{\textit{\ceur@conference}}%
    \fi
    \ifx\ceur@fntext\@empty\else
      \footnotetext{\textsuperscript{$\dagger$}\,\ceur@fntext}%
    \fi
    \ifx\ceur@emails\@empty\else
      \footnotetext{\ceur@emails}%
    \fi
    \ifx\ceur@copyrightclause\@empty\else
      \footnotetext{%
        \IfFileExists{cc-by.png}{%
          \raisebox{-0.35ex}{\includegraphics[height=2.2ex]{cc-by.png}}\,%
        }{}%
        \textcopyright{}%
        \ifx\ceur@copyrightyear\@empty\else\ \ceur@copyrightyear\fi
        \ \ceur@copyrightclause%
      }%
    \fi
  \endgroup
}
\makeatother

% Abstract / keywords / acknowledgments environments
\makeatletter
\RenewDocumentEnvironment{abstract}{+b}{\gdef\ceur@abstract{#1}}{}
\NewDocumentEnvironment{keywords}{+b}{\gdef\ceur@keywords{#1}}{}
\makeatother
\newcommand\sep{; }
\NewDocumentEnvironment{acknowledgments}{+b}{%
  \section*{Acknowledgments}#1%
}{}